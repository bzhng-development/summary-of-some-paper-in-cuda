# Llama 2: Open Foundation and Fine‑Tuned Chat Models

**ArXiv:** [2307.09288](https://arxiv.org/abs/2307.09288)
**Authors:** Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie‑Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom
**Institutions:** 

## 🎯 Pitch

Llama 2 presents an open, replicable framework for developing chat-optimized large language models, combining supervised fine-tuning with reinforcement learning from human feedback to enhance both helpfulness and safety. By bridging the gap between open and commercial models, it empowers the academic and open-source communities to improve and safely deploy AI assistants, paving the way for broader, impactful applications while maintaining transparency and reproducibility in AI alignment techniques.

---

## 1. Executive Summary (2–3 sentences)
Llama 2 introduces a fully open family of large language models and a chat-optimized variant, `Llama 2-Chat`, trained with a transparent and reproducible alignment recipe (supervised fine-tuning + reinforcement learning with human feedback). It narrows the gap between open models and commercial assistants by delivering strong helpfulness and safety, documented with extensive ablations, human studies, and safety evaluations, while also detailing practical alignment techniques (e.g., rejection sampling, PPO, Ghost Attention, targeted context distillation).

## 2. Context and Motivation
- Problem gap
  - Open pretrained models (e.g., BLOOM, LLaMA-1, Falcon) matched older closed models on raw capability but were not “product-ready” as assistants; the missing piece was scalable, well-documented alignment for dialogue helpfulness and safety (Introduction; Figures 1 and 3).
  - Alignment steps—especially data collection, reward modeling, and RLHF—are costly and opaque, limiting reproducibility and progress on safety research (Introduction).
- Why it matters
  - Real-world impact: chat assistants must be both helpful and safe; misaligned models can produce harmful or misleading content (Section 4).
  - Research significance: an open, end-to-end recipe for alignment (data guidelines, reward modeling decisions, RL choices, safety mitigations) enables community replication, critique, and improvement (Sections 3–4; Appendix).
- Prior approaches and limitations
  - Open models focused on pretraining quality but lacked transparent recipes for high-quality chat alignment at scale (Introduction; Section 3).
  - Closed models achieved strong helpfulness/safety but provided little methodological detail or training data openness, limiting scientific progress.
- Positioning
  - Llama 2 provides both pretrained models (7B, 13B, 70B) and chat models plus a detailed alignment pipeline (SFT → RM → RLHF via rejection sampling and PPO), with safety-specific data, evaluation, and red teaming (Sections 2–4). It also includes architectural updates for scalable inference (Grouped-Query Attention, doubled context length) and broad quantitative comparisons (Tables 3–4; Figures 1–3).

## 3. Technical Approach
This section explains the full pipeline, from pretraining to alignment and safety.

- Pretraining (Section 2)
  - Data and scale
    - 2 trillion tokens from publicly available sources; explicit removal of some high-PII sites; no Meta user data (Section 2.1).
    - Context length doubled to 4k tokens; vocabulary: 32k BPE (Section 2.2; Tokenizer).
  - Architecture choices (Section 2.2)
    - Transformer with RMSNorm, SwiGLU, and RoPE positional embeddings as in LLaMA-1.
    - New: `Grouped-Query Attention (GQA)` for larger models (34B, 70B) to reduce KV-cache memory and improve inference scalability (Table 1; Table 18; Figure 24).
      - Ablation: GQA performs on par with multi-head attention (MHA) and better than multi-query attention (MQA) on average (Table 18).
  - Training setup and footprint
    - AdamW, cosine LR, 2k warmup, weight decay 0.1, gradient clipping 1.0 (Section 2.2).
    - Carbon estimate: 539 tCO2eq for all pretraining runs, fully offset (Table 2).

- Supervised Fine-Tuning (SFT) for chat (Section 3.1)
  - Purpose: bootstrap alignment before RLHF using high-quality instruction-following examples.
  - Strategy: prioritize high-quality, diverse, dialogue-style human-written examples (27,540 total) over large third-party datasets; concatenate prompt/answer, mask user tokens, optimize on assistant tokens (Section 3.1; Table 5).

- Reinforcement Learning with Human Feedback (RLHF) (Section 3.2)
  - Step 1 — Human preference data (Section 3.2.1; Table 6)
    - Annotators write prompts, compare two model responses, and rate preference strength (“significantly better,” … “negligibly better/unsure”).
    - Coverage along two axes: helpfulness and safety; >1.4M binary comparisons collected in 14 batches (Table 26).
  - Step 2 — Reward models (RMs) (Section 3.2.2)
    - Two separate RMs: a `Helpfulness RM` and a `Safety RM`, each initialized from a chat model to keep knowledge aligned.
    - Core loss: pairwise ranking loss to score chosen higher than rejected response (Eq. (1)).
    - Innovation: add a `margin` term m(r) based on preference strength to push apart clearly different responses (Eq. (2); Table 27; ablation in Table 28 shows gains on “significantly better” pairs).
    - Training mix: combine Meta preference data with open datasets to improve generalization and reduce reward hacking (Section 3.2.2; Table 6).
    - Results: RMs outperform alternatives (SteamSHP-XL, OpenAssistant, GPT-4-as-a-judge) on internal helpfulness/safety tests (Table 7). Accuracy scales with data/model size (Figure 6).
  - Step 3 — Policy optimization (Section 3.2.3)
    - Two complementary algorithms:
      1) `Rejection Sampling fine-tuning`: sample K responses per prompt, score with the best RM, and fine-tune on the top-scoring responses (Figures 7–8).
         - Observations: larger K and appropriate temperature (e.g., T≈1.2–1.3 for later iterations) increase the “best-of-K” reward (Figure 8). Earlier versions experienced forgetting in niche capabilities; later versions mitigated by retaining top samples from earlier iterations (Section 3.2.3).
      2) `PPO` (Proximal Policy Optimization): optimize expected reward with a KL penalty to remain near the SFT policy (Eq. (3)–(4)). Uses a `piecewise reward`: Safety RM dominates for risky prompts or low safety scores; otherwise Helpfulness RM is used (Section 3.2.3).
         - Practicalities: use FSDP; consolidate weights before generation to avoid 20× slowdown (Section 3.2.3).

- Multi-turn consistency via `Ghost Attention (GAtt)` (Section 3.3)
  - Problem: chat models tend to forget system-level directives (e.g., “act as Oscar Wilde”) after several turns (Figure 9, left).
  - Method: create synthetic training where the instruction is prepended to each user turn during sampling, but compute loss only on assistant tokens of the final turn to avoid mismatch; fine-tune on these traces (Section 3.3).
  - Effect: sustained attention to the system message across many turns; attention visualizations show stronger focus on the instruction (Figure 10). Human eval shows 100% consistency up to 20 turns on tested attributes (Table 30).

- Safety alignment (Section 4)
  - Taxonomy and data (Section 4.2.1)
    - Risk categories: illicit/criminal, hateful/harmful, unqualified advice.
    - Attack vectors: psychological, logical, syntactic, semantic, perspective-based, non-English, etc.
  - Three techniques
    1) `Safety SFT`: adversarial prompts with safe demonstrations teach the model helpful but constrained responses (Section 4.2.2).
    2) `Safety RLHF`: targeted preference data for risky prompts; train a Safety RM and use it in rejection sampling + PPO (Section 4.2.3). Quantitatively improves the safety score distribution without reducing helpfulness (Figure 14).
       - Safety data scaling: adding more safety data markedly improves safety RM scores while keeping helpfulness stable (Figure 15).
       - False refusals: overall rare on standard helpfulness data (~0.05%) but more frequent on “borderline” prompts with sensitive tokens (Appendix Figure 33; Table 41).
    3) `Targeted context distillation`: prepend a “safety preprompt” (sometimes with category-specific answer template), generate, then fine-tune on the generated safe response without the preprompt (Section 4.2.4; Table 39).
       - Only keep distilled outputs when Safety RM score improves, avoiding generic or overly cautious answers (Figure 16).

## 4. Key Insights and Innovations
1) Dual reward models with margin-based ranking (Section 3.2.2; Eq. (2); Tables 7–8, 28)
   - What’s new: separate `Helpfulness RM` and `Safety RM` with a preference-strength-aware margin. This acknowledges and manages the tension between being helpful and being safe (Appendix A.4.1; Figure 32).
   - Why it matters: improves accuracy on clearly separable pairs (Table 28) and provides a robust, targeted signal for PPO and rejection sampling (Eq. (4)).

2) Iterative rejection sampling + PPO with temperature rescaling (Section 3.2.3; Figures 7–8, 21)
   - What’s new: a practical recipe that alternates best-of-K selection and PPO, with empirical guidance on sampling temperature, and evidence that RLHF adjusts the “effective temperature” by prompt type (Figure 21).
   - Why it matters: offers a reproducible path to steadily improve model quality while limiting reward hacking (KL penalty) and preventing regressions by retaining earlier high-quality samples.

3) `Ghost Attention (GAtt)` for multi-turn adherence to system instructions (Section 3.3; Figure 10; Table 30)
   - What’s new: a simple, data-only trick that stabilizes long-horizon obedience to a system message across 20+ turns; zero loss on intermediate turns avoids training mismatch.
   - Why it matters: multi-turn consistency is crucial for assistants; this method is lightweight, easy to reproduce, and demonstrably effective (100% in human checks, Table 30).

4) Targeted `context distillation` gated by Safety RM (Section 4.2.4; Figure 16; Table 40)
   - What’s new: use safety preprompts and category-specific answer templates to synthesize safer outputs, then keep only those that improve Safety RM scores.
   - Why it matters: improves safety on difficult adversarial prompts without broadly degrading helpfulness (Figure 16); also surfaces the trade-off of overly generic or falsely cautious responses and a way to mitigate it.

5) Scalable pretraining choices validated by ablations (Sections 2.2; A.2.1; Tables 16–18; Figure 24)
   - What’s new: doubled context length and GQA in larger models; systematic ablations show long-context gains and GQA’s speed/memory advantage with nearly unchanged accuracy.
   - Why it matters: practical, high-impact engineering decisions for real-world deployment.

## 5. Experimental Analysis
- Evaluation methodology
  - Base-model capability: grouped benchmark suite across code, commonsense, world knowledge, reading comprehension, math, MMLU, BBH, AGI Eval (Table 3); compared against LLaMA-1, MPT, Falcon; and to closed models (GPT-3.5, GPT-4, PaLM/PaLM-2) for selected tasks (Table 4).
  - Model-based validation during RLHF iterations: win rates vs. ChatGPT judged by their RMs and by GPT-4, to reduce in-house bias (Figure 11).
  - Human helpfulness evaluation: ~4,000 prompts (single and multi-turn), three raters each, 7-point scale, comparisons to open and closed models (Figure 12; Appendix A.3.7). Inter-rater reliability measured with Gwet’s AC2 (0.37–0.55 depending on comparison; Appendix).
  - Safety evaluation: ~2,000 adversarial prompts (single and multi-turn), five-point safety Likert scale, violation percentage (Figures 17–19). Automatic safety metrics: TruthfulQA and ToxiGen (Tables 11, 14). Red teaming with >350 participants across risk categories (Section 4.3).
  - Contamination analysis: suffix-array-based token-skipgram matching; flagged modest contamination effects primarily in HellaSwag and MMLU-Humanities (Appendix A.6; Table 51).

- Main quantitative results
  - Pretraining improvements (Table 3)
    - `Llama 2 70B` leads open-source base models on most aggregates; e.g., MMLU 68.9, BBH 51.2, AGI Eval 54.2.
    - Gains vs LLaMA-1 of similar size (e.g., +~5 points on MMLU, +~8 on BBH for 70B vs 65B).
  - Comparison to closed models (Table 4)
    - `Llama 2 70B` approaches GPT-3.5 on MMLU (68.9 vs 70.0) and GSM8K (56.8 vs 57.1), but lags on code (HumanEval 29.9 vs 48.1). Much lower than GPT-4/Palm-2-L on several tasks.
  - Helpfulness human study (Figure 12)
    - `Llama 2-Chat 70B` is competitive with ChatGPT: win 36%, tie 31.5% (loss ~32.5%) on their ~4k prompt set.
    - Strong wins vs open models: e.g., 34B chat variant >75% win rate vs Vicuna-33B and Falcon-40B.
  - Model-based progression (Figure 11)
    - Iterative RLHF improves win-rate vs ChatGPT; with GPT-4 as judge, latest version exceeds 60% win rate on helpfulness.
  - Safety (Figures 17–19; Table 14)
    - Violation rates: `Llama 2-Chat` has low overall violation percentage across sizes, competitive with or below many baselines (Figure 17a).
    - On TruthfulQA + ToxiGen: `Llama 2-Chat 70B` achieves 64.14% (truthful+informative) and essentially 0.01% toxic outputs—lowest toxicity among compared models (Table 14).
    - Multi-turn prompts induce more violations across all models, but `Llama 2-Chat` degrades less than some baselines (Figure 18).
  - Reward models (Tables 7–8; Figure 6)
    - Both RMs beat public baselines and GPT-4 on Meta test sets; accuracy scales with data/model size and is highest when responses differ more strongly (“significantly better”).
  - Safety ablations (Figure 15; Appendix)
    - Increasing safety data improves the safety score distribution’s tail with stable helpfulness means (Figure 15).
    - False refusal remains low on standard helpfulness data (~0.05% at most) but higher on a curated borderline set (Appendix Figure 33; examples in Table 41).
  - Pretraining ablations and efficiency (Tables 16–18; Figure 24)
    - 4k context helps long-context tasks with no loss on general tasks (Table 16–17).
    - GQA yields throughput gains with similar latency and accuracy to MHA, superior to MQA (Table 18; Figure 24).
  - Qualitative observations (Section 5.1; Figures 21–23; Table 15)
    - “In-context temperature rescaling”: RLHF reduces creative diversity less than factual diversity as temperature increases (Figure 21).
    - Emergent tool use: with zero explicit tool-use training, `Llama 2-Chat` can chain tools in a prompt (Figure 23) and substantially improves on tool-use math datasets with a calculator (Table 15).

- Do the experiments support the claims?
  - Yes, with caveats. The paper provides:
    - Broad quantitative evidence (Tables 3–4, 11, 14) and careful human studies (Figure 12) showing that `Llama 2-Chat` is competitive with some closed assistants and ahead of open baselines on their prompt set.
    - Robust safety improvements via SFT + RLHF + targeted distillation (Figures 14–19).
    - Clear ablations for key design choices (GQA, context length; Table 18; Tables 16–17) and detailed RM scaling (Figure 6).
  - Caveats
    - Human evaluations have subjectivity and prompt-set bias (Figure 12 caveats; Section 3.4.2).
    - Some benchmarks (e.g., HellaSwag, MMLU-Humanities) show signs of contamination effects (Appendix Table 51).
    - Performance vs GPT-4 and PaLM-2-L remains substantially lower on several hard tasks (Table 4).

## 6. Limitations and Trade-offs
- Scope and assumptions
  - Language skew: training and tuning are primarily English; performance in other languages is “fragile” (Section 5.2; Table 10).
  - Safety content standards are tuned to their guidelines and may bias evaluations toward their model’s behavior (Figure 3 caption; Section 4.4).
- Safety–helpfulness tension and false refusals
  - While mean helpfulness holds steady as safety data scales (Figure 15), the rate of false refusals increases on a deliberately “borderline” test (Appendix Figure 33). Some context-distilled responses become vague or overcautious (Table 40).
- Capability gaps
  - Code generation lags GPT-3.5 and GPT-4 (HumanEval numbers in Table 4).
  - Multi-turn safety is harder for all models; though Llama 2-Chat degrades less, it still degrades (Figure 18).
- Resource and environmental costs
  - Training uses 3.3M GPU-hours on A100-80GB; 539 tCO2eq (Table 2). Although offset, this is non-trivial.
- Data and evaluation concerns
  - Modest contamination for some evals (Appendix Table 51).
  - Human helpfulness IRR is moderate (AC2 ~0.37–0.55), typical but indicative of noise (Section 3.4.2).
- Algorithmic trade-offs
  - Reward models favor their data distribution; despite mixing open datasets, Goodhart risks remain (Section 3.4.1).
  - Rejection sampling improves maxima at the cost of more sampling compute; correct temperature is iteration dependent (Figure 8).

## 7. Implications and Future Directions
- Field impact
  - Provides a practical, open recipe for building helpful and safe assistants: SFT → dual RMs with margin → rejection sampling + PPO → safety GAtt + targeted context distillation. This directly elevates the baseline for open LLM alignment research.
  - Engineering guidance (GQA, longer context) and thorough ablations help practitioners balance scalability and quality.
- Follow-up research enabled
  - Better multi-objective reward modeling beyond two separate RMs; methods that explicitly handle safety–helpfulness trade-offs and calibration (Appendix A.4.1).
  - Improved methods to reduce false refusals without sacrificing safety—e.g., dynamic, risk-aware preprompting or selective refusal classifiers integrated at inference.
  - Richer red teaming and multilingual safety; more realistic, task-oriented multi-turn evaluations (Section 3.4.2 limitations).
  - System-message control: extend `Ghost Attention` to allow controlled switching of personas/goals mid-dialogue (Section 3.3).
  - Benchmarks: reduce contamination, broaden content types (reasoning/coding), and include end-to-end task success.
- Applications
  - Foundation for safe, helpful chat systems in consumer products and enterprise settings.
  - Educational assistants with strong refusal behavior on unsafe queries but contextual guidance to safe, useful alternatives (Sections 4.2–4.4).
  - Research testbed for RLHF methods, safety taxonomies, and reward-model diagnostics at scale.

> Representative claims and results cited
> - “Llama 2-Chat 70B… win rate of 36% and tie rate of 31.5% relative to ChatGPT” (Figure 12).
> - “Percentage of toxic generations shrinks to effectively 0%” for `Llama 2-Chat` across sizes (Table 14).
> - “Safety RLHF… shifts safety RM scores upward without hurting helpfulness” (Figure 14).
> - “GQA” offers higher throughput with near-par accuracy vs MHA and better than MQA (Table 18; Figure 24).
> - Dual RMs with margin improve ranking accuracy on distinct pairs (Table 28); overall RM performance scales with data/model size (Figure 6).

In sum, this work contributes a replicable alignment blueprint, safety-focused innovations (GAtt, targeted distillation), and evidence that open chat models can reach competitive helpfulness and strong safety—while candidly documenting remaining gaps, trade-offs, and the road ahead.
