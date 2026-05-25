# Gemma 2: Improving Open Language Models at a Practical Size

**ArXiv:** [2408.00118](https://arxiv.org/abs/2408.00118)

## 🎯 Pitch

Gemma 2 debuts a new wave of open-weight language models (2B, 9B, and 27B parameters) that match or surpass larger competitors by training smaller models on extensive teacher-generated knowledge through large-scale distillation, coupled with Transformer architectural improvements. This innovative approach delivers state-of-the-art results at a practical scale—making high-quality language models accessible for organizations and researchers who need efficient, reliable deployment without the prohibitive costs of massive models.

---

## 1. Executive Summary (2-3 sentences)
Gemma 2 introduces a family of open-weight language models at 2B, 9B, and 27B parameters that reach state-of-the-art quality for their size by combining large-scale knowledge distillation with targeted Transformer design changes. The key advance is training the 2B and 9B “student” models on huge amounts of teacher-generated signal—far beyond compute-optimal token counts—plus architectural and post-training choices that together yield performance competitive with models 2–3× larger (e.g., Table 12, Table 14).

## 2. Context and Motivation
- Problem addressed
  - Small open models have improved mainly by training on ever more raw tokens (“longer training”). This strategy yields diminishing returns because performance scales only logarithmically with dataset size, so each additional trillion tokens produces smaller gains (Introduction; citing Hoffmann et al., 2022). Recent small models report up to 15T tokens to gain <1–2% (Introduction).
  - Many small models are still “under-trained” in the sense that their learning signal (one-hot next-token targets) is sparse and uninformative.

- Why it matters
  - Practical deployment favors small models due to memory and latency constraints. Achieving large-model quality at small-model sizes lowers cost and broadens access for research and products.

- Prior approaches and their limits
  - Next-token prediction on raw text dominates; it uses a one-hot target that conveys no graded information about plausible alternatives. This under-utilizes each training step’s potential gradient signal.
  - Scaling model size or token count alone is expensive and yields diminishing returns.

- How Gemma 2 positions itself
  - Replaces the sparse one-hot label with a dense probability distribution from a strong “teacher” model via knowledge distillation, improving the quality of supervision at every token (Section 3.2).
  - Marries this with practical architectural choices (interleaved local/global attention, Grouped-Query Attention, robust normalization, logit soft-capping) and a refined post-training pipeline (SFT, RLHF, model averaging) to systematically lift small-model performance (Sections 2 and 4).

## 3. Technical Approach
Gemma 2 covers three model sizes—`2B`, `9B`, `27B`—with a shared architecture and size-specific hyperparameters (Table 1).

A. Architecture (Section 2; Table 1–2)
- Core backbone
  - Decoder-only Transformer with context length 8192 and Rotary Position Embeddings (`RoPE`), and GeGLU feed-forward layers (Section 2; Table 1).
- Local + global attention interleaving
  - Every other layer alternates between:
    - Local sliding-window self-attention over the most recent 4096 tokens.
    - Global attention over the full 8192-token span.
  - Quote: 
    > “We alternate between a local sliding window attention … sliding window size … 4096 tokens, while the span of the global attention layers is set to 8192 tokens.” (Section 2)
  - Why: Local attention reduces compute and memory for long contexts; periodic global layers refresh global connectivity to prevent drift.
- Grouped-Query Attention (`GQA`) with `num_groups=2`
  - Definition: GQA ties multiple query heads to a smaller set of key/value heads, reducing KV parameters and speeding inference with minimal quality loss.
  - Evidence: A 9B ablation shows essentially unchanged accuracy vs. full Multi-Head Attention (MHA): average over 4 benchmarks 50.8 (GQA) vs. 50.3 (MHA) (Table 8). Chosen for efficiency at deployment time.
- Normalization and stability
  - Both pre-norm and post-norm using `RMSNorm` around attention and feed-forward sublayers to stabilize deep training (Section 2).
- Logit soft-capping
  - Definition: Clamps logits with `logits ← soft_cap * tanh(logits/soft_cap)` to keep them within [-soft_cap, +soft_cap].
  - Settings: soft_cap=50.0 for self-attention layers; 30.0 for the final layer (Section 2). Purpose: prevents unstable extremes and improves optimization robustness.
- Vocabulary/embeddings
  - 256k-token SentencePiece tokenizer shared with Gemini/Gemma 1 (Section 3.1). This choice increases embedding parameters (Table 2) but supports many languages at the tokenization level.

B. Pre-training (Section 3)
- Data scale and scope
  - `27B`: 13T tokens; `9B`: 8T tokens; `2B`: 2T tokens. Sources: web documents, code, science; mostly English (Section 3.1).
  - Filtering to reduce unsafe content, sensitive data, and benchmark contamination (Section 3.1).
- Knowledge distillation objective (Section 3.2)
  - Definition: Instead of learning from one-hot next-token targets, the student matches the teacher’s full next-token distribution `P_T(x | x_c)` by minimizing cross-entropy:
    > min over P_S: sum_x [ - P_T(x | x_c) * log P_S(x | x_c) ] (Section 3.2)
  - Intuition: The probability mass on near-miss tokens carries rich information (“dark knowledge”), yielding better gradients per step.
  - Strategy: Train small models (2B, 9B) with distillation on far more tokens than compute-optimal scaling would prescribe—“more than 50×” the predicted compute-optimal quantity (Introduction). The 27B model is trained from scratch on 13T tokens (Section 3.1).
- Compute infrastructure (Section 3.3; Table 3)
  - TPU pods (v4, v5e, v5p) with data/model sharding (GSPMD) and ZeRO-3-like optimizer partitioning; cross-replica reductions via Pathways (Table 3).
- Carbon accounting (Section 3.4)
  - Estimated pre-training emissions: 1247.61 tCO2e; Google data centers are carbon-neutral via renewable energy and offsets (Section 3.4).

C. Post-training (Instruction Tuning) (Section 4; Tables 4–5)
- Supervised fine-tuning (SFT)
  - Mix of human- and teacher-generated prompt–response pairs (predominantly synthetic), plus “distillation from the teacher on the student’s distribution” during SFT (Section 4).
- RLHF
  - Reward model “an order of magnitude larger than the policy” and oriented to multi-turn dialogue; policy optimized on the same prompts as SFT (Section 4).
- Model merging
  - Averages checkpoints trained with different hyperparameters (“Warp”-style weight averaging) to improve overall quality (Section 4; cites Ramé et al., 2024).
- Safety data curation and formatting
  - Filtering synthetic data for PII, toxicity, self-identification mistakes, deduplication; include data that encourages attribution, hedging, and refusals to reduce hallucinations (Section 4).
  - Chat format uses explicit control tokens (Table 4) and a new schema where generations end with `<end_of_turn><eos>` (Table 5). This improves multi-turn handling by clearly marking turn boundaries.

## 4. Key Insights and Innovations
1) Large-scale knowledge distillation for small models (fundamental)
- What’s new
  - Uses a powerful teacher to generate dense supervision on every token and trains `2B` and `9B` students on token counts vastly exceeding compute-optimal predictions (Introduction; Section 3.2).
- Why it matters
  - Substantially boosts small-model performance without increasing model size; simulates training “beyond the number of available tokens” by extracting more learning signal per token (Section 3.2).
- Evidence
  - Training 2B on 500B tokens: distillation improves average score from 60.3 to 67.7 across three benchmarks (Table 6).
  - Perplexity reductions with distillation hold as model size increases: e.g., at 1B parameters, PPL 17 (distilled) vs. 17 (from scratch) — consistent improvement at smaller scales (Table 7).

2) Practical long-context attention via interleaved local and global attention (incremental but impactful)
- What’s different
  - Alternating local (4096 window) and global (8192 span) attention reduces cost while keeping periodic global connectivity (Section 2).
- Why it matters
  - Enables 8K contexts at moderate compute; at inference, the local window can be shrunk with only modest perplexity impact:
    > 9B PPL: 1.63 (4096), 1.63 (2048), 1.64 (1024) (Table 10).

3) Inference-friendly attention (GQA) without accuracy loss (incremental)
- What’s different
  - `GQA` with 2 groups reduces KV parameters and speeds decoding compared to standard MHA (Section 2).
- Evidence
  - 9B average across 4 benchmarks: 50.8 (GQA) vs. 50.3 (MHA) (Table 8).

4) Robust training stack for stability and performance (incremental ensemble)
- Elements
  - Logit soft-capping, RMSNorm both pre- and post- sublayer, deeper networks favored over wider at fixed parameter count (Section 2).
- Evidence
  - “Wide vs deep” ablation on 9B: deep model averages 52.0 vs. 50.8 over 4 benchmarks (Table 9).

5) Post-training with larger, multi-turn-oriented reward models and model averaging (incremental but effective)
- What’s different
  - Reward model much larger than policy; multi-turn oriented; checkpoint averaging to combine strengths (Section 4).
- Why it matters
  - Improves instruction-following, safety, multi-turn quality (Tables 15–16) and pushes real-world chat quality (Table 14).

## 5. Experimental Analysis
A. Evaluation methodology
- Pretraining evaluations (Section 6.1)
  - Compare Gemma 2 `27B` (from scratch) to similarly sized or larger open models on standard benchmarks (MMLU, GSM8K, ARC-c, HellaSwag, Winogrande) with the HuggingFace evaluation suite (Table 12).
- Post-training evaluations (Section 6.2)
  - Human preference via LMSYS Chatbot Arena Elo (blind, side-by-side; Table 14).
  - Internal human studies on instruction following, safety, and multi-turn conversation quality (Tables 15–16).
  - Few-shot benchmarks before/after instruction tuning (MMLU, MBPP; Table 17).
  - Safety and responsibility evaluations: toxicity/bias/factuality benchmarks (Table 18), “dangerous capabilities” assurance evaluations (Tables 19–21), and persuasion studies (Tables 22–25).
- Ablations and robustness
  - Distillation vs. from-scratch (Table 6).
  - Distillation effect across model sizes (Table 7).
  - GQA vs. MHA (Table 8).
  - Wide vs. deep (Table 9).
  - Sliding-window size at inference (Table 10).
  - Prompt formatting robustness (Table 11).

B. Main quantitative results
- Small models trained with distillation perform far better than from-scratch
  - 2B trained on 500B tokens: +7.4 points average across 3 benchmarks with distillation (Table 6).
  - Perplexity decreases consistently with distillation across sizes (Table 7).
- 27B pre-trained competes with much larger models
  - On HuggingFace suite: `27B` achieves MMLU 75.2 vs. LLaMA-3 70B at 79.2; GSM8K 74.0 vs. 76.9; ARC-c 71.4 vs. 68.8; HellaSwag 86.4 vs. 88.0; Winogrande 83.7 vs. 85.3 (Table 12). It outperforms Qwen1.5 32B on most metrics (Table 12).
- Across 2B–27B, Gemma 2 improves substantially over prior open baselines at similar size
  - Example (Average over 8 common benchmarks): `9B` scores 70.2 vs. Mistral 7B 61.0 and LLaMA-3 8B 61.9; `27B` reaches 74.4 (Table 13).
- Human preference (arena)
  - Elo: `27B-IT` 1218 (higher than LLaMA-3 70B Instruct at 1206), `9B-IT` 1187 (roughly GPT-4-0314 at 1186), `2B-IT` 1126 (higher than GPT-3.5-Turbo-0613 at 1116) (Table 14).
- Instruction following and safety (internal human evals)
  - Safety win-loss vs GPT-4o: `2B-IT` 53% wins / 38% losses; `9B-IT` 48.2% / 28.3%; `27B-IT` 49.6% / 39.6% (Table 15).
  - Instruction following (single-sided): rises from 24.3% (Gemma 1.1 7B) to 34.1% (`9B-IT`) and 37.7% (`27B-IT`) (Table 15).
- Multi-turn user studies
  - User satisfaction (1–5): `7B-1.1` 3.32 → `2B-IT` 3.64 → `9B-IT` 4.04 → `27B-IT` 4.20; similar trend for “goal achievement” (Table 16).
- Instruction tuning improves few-shot scores
  - `2B`: MMLU 52.2 → 56.1; MBPP 30.2 → 36.6. `27B`: MMLU 75.2 → 76.2; MBPP 62.6 → 67.4 (Table 17).
- Memorization and privacy
  - Overall memorization rates are <0.1% and lower than prior models; approximate memorization is higher than exact but still low (Figure 1). Found no high-severity personal data emissions; low rate (0.00026%) of lower-severity PII among memorized outputs (Section 7).
- Safety/assurance evaluations (Tables 18–21)
  - Toxicity/bias: mixed but generally strong improvements vs. Gemma 1.1. Example: TruthfulQA MC2Acc `27B-IT` 51.60 (Table 18).
  - Offensive cybersecurity: `27B` solves 34/76 InterCode-CTF tasks, a notable jump over CodeGemma 7B 12/76 but below Gemini 1.5 Pro 62/76 (Table 19).
  - Vulnerability detection: near-chance on several datasets; similar to Gemini 1.0 Ultra on SecretPatch (Table 20).
  - Self-proliferation: completes more milestones than Gemini 1.0 Ultra but passes 0/10 end-to-end tasks (Table 21).
- Persuasion studies (Tables 22–25)
  - Rapport-building is strong (e.g., 80% report “personal connection”), but deceptive persuasion is limited and not significantly stronger than Gemini baselines (Tables 22–25).

C. Do the experiments support the claims?
- The ablations (Tables 6–11) directly support the central claim: distillation materially improves small models and the chosen architectural tweaks are accuracy-neutral or modestly positive while improving efficiency.
- Reported head-to-heads on widely used benchmarks (Tables 12–13) and human preference ratings (Table 14) align with the claim of state-of-the-art performance for the parameter ranges.
- Safety and memorization sections (Figure 1; Table 18) show measured, not asserted, behavior; assurance evals (Tables 19–21) honestly reveal limited dangerous capabilities, matching the paper’s caution.

D. Robustness checks and caveats
- Formatting robustness: Gemma 2 models show relatively low variance; e.g., Gemma 2 9B std-dev 0.9 on MMLU vs. Mistral 7B at 6.9 (Table 11).
- Inference-time sliding-window reductions barely affect perplexity (Table 10), indicating resilience to smaller windows.
- Mixed results exist (e.g., near-chance code vulnerability detection; Table 20), correctly signaling task boundaries.

## 6. Limitations and Trade-offs
- Dependence on a strong teacher
  - Distillation quality is bounded by the teacher’s quality and biases; students may inherit teacher errors or stylistic artifacts (implicit in Section 3.2).
- Data scope and multilinguality
  - Training is primarily English and not multimodal (Section 3.1). Although the tokenizer is multilingual-friendly (256k vocab), no claim of SOTA multilingual ability is made.
- Compute and data demands
  - While model sizes are modest, training uses very large token counts and large TPU pods (Section 3.3; Table 3). Distillation at this scale still requires substantial infrastructure.
- Benchmark sensitivity and format variance
  - Although improved, smaller models are less robust to formatting than larger ones (e.g., 2B std-dev 2.1 vs. 9B 0.9; Table 11).
- Limited “extreme capability”
  - Assurance evaluations show low end-to-end offensive cybersecurity and self-proliferation capabilities (Tables 19–21), which is desirable for safety but also indicates limits on complex autonomous tool use.
- Not all domains benefit equally
  - Some code vulnerability tasks remain near chance (Table 20). Gains vary by task type: e.g., MATH lags larger closed models despite improvements (Table 13).
- Architectural choices trade interpretability for efficiency
  - GQA ties KV heads, which can slightly constrain representational flexibility; chosen because it is accuracy-neutral in ablations (Table 8) but that may not generalize to all task distributions.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that small, open models can achieve near–large model quality by upgrading supervision quality (distillation) rather than only scaling size or tokens. This lowers the barrier for high-quality local deployment and research replication (Tables 12–14).
- Research enabled or suggested
  - Distillation-at-scale design space:
    - Teacher choice, temperature schedules, and confidence calibration to maximize student learning.
    - Mixtures of teachers or task-specialized teachers.
    - Extending to multimodal or multilingual teachers/students.
  - Architectural extensions:
    - Dynamic schedules for local/global layer frequency.
    - Adaptive sliding-window sizes conditioned on load or content (given Table 10’s robustness).
    - Further KV-cache compression leveraging GQA-like ideas.
  - Post-training science:
    - Systematic study of reward-model scale vs. quality (Section 4 mentions reward model >10× policy).
    - Model merging strategies (e.g., layer-wise or task-wise averaging) and their theory (Section 4).
  - Safety and reliability:
    - Improving factuality via in-context attribution/hedging datasets (Section 4).
    - Better defenses against approximate memorization and more nuanced PII detection, given low but non-zero rates (Section 7; Figure 1).
- Practical applications
  - On-device assistants and enterprise deployments that need strong quality under strict latency/memory budgets (benefiting from GQA, long context, and high Elo scores; Tables 1, 10, 14).
  - Coding, science QA, and reasoning assistants where the `9B`/`27B` models already show strong benchmark performance (Table 13).
  - Multi-turn conversational agents with improved adherence and safety (Tables 15–16), using the provided chat formatting (Tables 4–5).

In short, Gemma 2’s main contribution is to turn “better supervision per token” into real, measurable small-model gains. The combination of scaled distillation, pragmatic attention design, and careful post-training offers a reproducible recipe for practical, capable, and safer open models.
