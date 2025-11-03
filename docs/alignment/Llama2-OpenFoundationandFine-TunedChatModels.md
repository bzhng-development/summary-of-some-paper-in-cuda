# Llama 2: Open Foundation and Fine-Tuned Chat Models

**ArXiv:** [2307.09288](https://arxiv.org/abs/2307.09288)

## 🎯 Pitch

Llama 2 introduces a family of large, open-source language models (7B–70B parameters) and a dialogue-optimized variant, Llama 2-Chat, developed with a transparent, reproducible pipeline for supervised fine-tuning, RLHF, and safety alignment. By openly detailing best practices for aligning LLMs with human preferences and safety, Llama 2 bridges the performance gap between open and commercial chat assistants, empowering the research and developer community to build safer and more effective AI applications without reliance on proprietary systems.

---

## 1. Executive Summary
Llama 2 is an openly released family of large language models (7B–70B parameters) plus a dialogue-optimized variant, `Llama 2-Chat`, trained with a transparent fine‑tuning and safety pipeline. It addresses the gap between open pretrained models and “product-grade” chat assistants by detailing a reproducible approach—high‑quality supervised fine‑tuning (SFT), iterative reinforcement learning from human feedback (RLHF), targeted safety methods, and multi‑turn consistency—while achieving strong helpfulness and safety relative to open and some closed baselines (Figures 1–3, 12; Tables 3–4, 14).

## 2. Context and Motivation
- Problem addressed
  - Open communities had strong pretrained LLMs (e.g., LLaMA‑1, BLOOM, Falcon) but lacked transparent, reproducible processes to reach “assistant-grade” chat quality and safety comparable to closed models like ChatGPT/Claude (Section 1). As a result, open models often underperformed in real dialogue use cases.

- Importance
  - Practical: High‑quality, safe assistants increasingly power consumer, enterprise, and research tools. Having an open, replicable method reduces dependence on closed systems and expands responsible innovation.
  - Scientific: RLHF, safety alignment, and multi‑turn control techniques had been either proprietary or under-documented; more transparency is needed to advance alignment research.

- Prior approaches and their shortcomings
  - Pretrained open models matched earlier closed pretrained models (e.g., LLaMA‑1 near GPT‑3), but were not competitive with heavily aligned chat assistants because the data, RLHF details, and safety tuning were costly or opaque (Section 1).
  - Alignment recipes existed (e.g., InstructGPT, Constitutional AI), but practitioners lacked a full, reproducible end‑to‑end blueprint tied to open models and code.

- Positioning of this work
  - Provides an open, end‑to‑end blueprint: data composition, training details, RLHF (with both rejection sampling and PPO), separate reward models for helpfulness and safety, iterative collection to keep rewards on‑distribution, multi‑turn control (`Ghost Attention`), and targeted safety techniques (Section 3–4).
  - Releases performant models up to 70B parameters, demonstrating competitive helpfulness and substantially improved safety (Figures 1–3, 12, 17–19; Table 14).

## 3. Technical Approach
This section explains the model family, the training pipeline, and the safety and multi‑turn control methods.

### 3.1 Pretraining (Section 2)
- Model family
  - `Llama 2` base models: 7B, 13B, 34B (not released), 70B parameters (Table 1).
  - Context window doubled to 4k tokens (vs. 2k in LLaMA‑1), improving long‑context tasks (Table 16) without hurting general tasks (Table 17).
  - Uses `Grouped-Query Attention (GQA)` for 34B/70B to reduce KV‑cache memory and improve inference scalability while keeping quality close to standard multi‑head attention; GQA outperforms single‑KV `MQA` and is near `MHA` (Table 18; Figure 24 shows throughput scaling).

- Data and optimization
  - 2 trillion tokens of “publicly available” data; more robust cleaning; upsampling factual sources (Section 2.1).
  - Tokenizer: same 32k BPE as LLaMA‑1.
  - Optimization: AdamW with cosine LR schedule; warmup 2000 steps; weight decay 0.1; gradient clip 1.0 (Section 2.2).
  - Training did not saturate at 2T tokens (Figure 5).

- Infrastructure and carbon accounting
  - Trained on NVIDIA A100s across two clusters (InfiniBand and RoCE); 3.3M GPU hours; 539 tCO2eq estimated emissions, fully offset (Table 2; Section 2.2.1).

- Base model evaluation
  - Across standard benchmarks (code, commonsense, world knowledge, reading, math, MMLU/BBH), Llama 2 improves over LLaMA‑1 and other open base models (Table 3), and is close to GPT‑3.5 on MMLU and GSM8K though behind on code (Table 4).

### 3.2 Supervised Fine‑Tuning (SFT) (Section 3.1)
- Goal
  - Bootstrap instruction-following and safe refusal behavior before RLHF.

- Data strategy
  - Start from public instruction tuning data, then emphasize quality over quantity: 27,540 curated SFT annotations collected via vendors with strict QA (Section 3.1). Example SFT items show helpful content and safety‑aware refusals (Table 5).

- Training setup
  - Cosine LR; initial LR 2e‑5; weight decay 0.1; batch size 64; sequence length 4096; backprop only on assistant tokens; 2 epochs (Section 3.1).

### 3.3 RLHF (Section 3.2)
Definitions:
- `RLHF` aligns a model to human preferences by training a `reward model (RM)` from human comparisons and then optimizing the policy to increase reward.
- `Helpfulness` vs. `Safety`: Separate signals because maximizing one can harm the other (empirically demonstrated later).

Pipeline steps:
1. Human preference data (Section 3.2.1)
   - Annotators write prompts, then pick the preferred response between two model outputs and label strength: “significantly better, better, slightly better, negligibly better/unsure.”
   - Separate helpfulness and safety collections; safety includes a label indicating if picked response is safe and whether the other is safe/unsafe (three bins).
   - Large-scale data: 1.42M comparisons from Meta plus curated open datasets (Table 6). Detailed weekly growth shows longer, multi‑turn data over time (Table 26; Figure 25).

2. Reward modeling (Section 3.2.2)
   - Two RMs trained from chat-model checkpoints: a `Helpfulness RM` and a `Safety RM` to avoid interference between objectives.
   - Loss: binary ranking loss with an additive `margin` that depends on rater confidence (Eq. 2; Table 27). Margin increases separation on clearer comparisons, boosting accuracy where differences are strong (Table 28; Figure 27 shows distribution shift).
   - Data mixing recipes:
     - Helpfulness RM: all Meta Helpfulness + a balanced mixture from Meta Safety and open datasets.
     - Safety RM: all Meta Safety + Anthropic Harmless + 10% helpfulness data to better resolve “both responses are safe” cases (Section 3.2.2).
   - Results and scaling:
     - RMs outperform baselines, including GPT‑4, on internal tests (Table 7). Accuracy is highest on “significantly better” pairs and falls as pairs become more similar (Table 8).
     - Accuracy improves with both data volume and model size and does not saturate (Figure 6).

3. Policy optimization (Section 3.2.3)
   - `Rejection Sampling fine-tuning`: For each prompt, sample K responses from the current model, score with the best RM, select the top response, then fine‑tune on this “best of K.” Figure 7 shows the gap between max and median reward grows with K, indicating headroom from exploration. Figure 8 shows optimal sampling temperature shifts after RLHF, especially at larger K.
   - `PPO` (Proximal Policy Optimization): Optimize expected reward with a KL penalty to the SFT policy to avoid reward hacking (Eq. 3–4). The immediate reward `Rc` uses Safety RM for risky prompts or low safety scores, else Helpfulness RM; scores are whitened to stabilize PPO. Hyperparameters: β=0.01 (7B/13B), 0.005 (34B/70B); batch 512; 200–400 iterations (Section 3.2.3).
   - Iterative strategy: multiple RLHF rounds (V1–V5). To prevent forgetting observed in V3 (e.g., loss of rhyme skill), later rounds retain top samples from earlier rounds (Section 3.2.3).

### 3.4 Multi‑turn Control with `Ghost Attention (GAtt)` (Section 3.3)
- Problem
  - Given an instruction that should persist across turns (e.g., “always answer with emojis” or “act as X”), early chat models forgot it after a few turns (Figure 9 left).

- Method
  - Create synthetic multi‑turn contexts where the user messages are augmented with the system instruction; sample model replies; then fine‑tune on these samples while zeroing the loss for all prior turns to avoid mismatch (Section 3.3).
  - During fine‑tuning, only the last assistant turn contributes to loss; the latent attention is encouraged to keep referencing the system instruction across the dialogue.

- Evidence
  - Human evaluation shows 100% adherence up to 20 turns for tested attributes vs. rapid decay without GAtt (Table 30).
  - Attention maps reveal persistent high activation on the system message with GAtt (Figure 10).

### 3.5 Safety Methodology (Section 4)
Definitions:
- `False refusal`: a refusal of a benign prompt due to misclassification as unsafe (Section 4.2.3).

Components:
1. Safety problem framing:
   - Risk categories: illicit/criminal, hateful/harmful, and unqualified advice (Section 4.2.1).
   - Attack vectors: psychological, logical, syntactic, semantic, perspective manipulations, non‑English, etc.

2. Safety SFT:
   - Adversarial prompts + high‑quality safe demonstrations teach refusal structure and safer alternatives (Section 4.2.2).

3. Safety RLHF:
   - Train a dedicated Safety RM on adversarial preference data and use the same RLHF machinery (Section 4.2.3).
   - Distribution shifts show improved safety without hurting helpfulness (Figure 14). Table 12 gives a qualitative example (scam email: unsafe SFT‑v2 vs. safe RLHF‑V5).

4. Safety data scaling:
   - Increasing safety data from 0% to 100% (with ~0.9M helpfulness samples fixed) shifts the long tail of unsafe responses upward while keeping average helpfulness stable (Figure 15 left/right). This quantifies the benefit of extra safety data.

5. Targeted `Context Distillation` for safety:
   - Prepend safety preprompts (generic or category‑specific with answer templates) to generate a safer answer, then fine‑tune on that output without the preprompt—distilling the safety context into the model (Section 4.2.4; Table 13).
   - A Safety RM gate keeps only distillations that increase safety score, avoiding vague or over‑cautious responses (Figure 16).

6. Red teaming:
   - >350 participants across domains; iterative cycles reduced “violating prompts per person-hour” γ from 1.8 → 0.45 and achieved ~90% mitigation of previously found issues across model versions (Section 4.3).

7. Measuring false refusals:
   - Extremely low on helpfulness data (~0.05% at 100% safety data) but higher on a curated “borderline” benign set with sensitive words (15–27%, Figure 33; examples in Table 41).

### 3.6 Additional observations (Section 5)
- “In‑context temperature rescaling”: after RLHF, factual prompts show reduced diversity at higher temperatures while creative prompts retain diversity (Figure 21).
- Temporal awareness from limited SFT (1,000 date‑focused examples) yields consistent date‑relative answers (Figure 22).
- Zero‑shot tool use emergence (e.g., chaining SEARCH and CALCULATOR tools) despite no explicit tool-use training; with a calculator, `Llama 2-Chat` outperforms Toolformer and others on math word‑problem datasets (Table 15; Figure 23).

## 4. Key Insights and Innovations
1. Separate `Helpfulness` and `Safety` reward models at scale (Section 3.2.2)
   - What’s new: Two specialized RMs initialized from chat checkpoints, with a preference‑margin loss and carefully mixed training subsets.
   - Why it matters: Resolves the empirically observed tension (Figure 32), improves accuracy on distinct pairs (Table 8), and allows targeted safety optimization without collapsing helpfulness (Figure 14, Figure 15).

2. Iterative RLHF with `Rejection Sampling` + `PPO` and temperature-aware exploration (Section 3.2.3)
   - What’s new: Combines best‑of‑K ranking (for breadth) with PPO (for depth), and shows that optimal sampling temperature shifts after RLHF (Figure 8).
   - Why it matters: Yields stable, higher‑quality policy updates while mitigating forgetting by reusing strong past samples; supports efficient exploration (Figure 7).

3. `Ghost Attention (GAtt)` for multi‑turn instruction persistence (Section 3.3)
   - What’s new: A simple, data‑level technique: augment system instructions across turns during sampling, then fine‑tune on the final turn with zero loss on earlier turns.
   - Why it matters: Achieves robust long‑turn consistency (100% adherence up to 20 turns, Table 30) and visibly reorients attention to the system message (Figure 10). This is a practical, low‑overhead alternative to architectural changes.

4. Targeted `Safety Context Distillation` gated by a Safety RM (Section 4.2.4)
   - What’s new: Use generic or category‑tailored preprompts to elicit safer answers; keep only distillations that improve Safety RM scores.
   - Why it matters: Efficiently improves safety on hard prompts (Figure 16a) while avoiding degradation on already‑good responses (Figure 16b). This addresses a common failure mode of naive preprompt distillation (Table 40).

5. Transparent, end‑to‑end open recipe (Sections 2–4, Appendix)
   - What’s new: Full disclosure of data composition trends, RM mixing, loss functions, hyperparameters, ablations (context length, GQA vs. MQA/MHA), safety scaling, and contamination analysis.
   - Why it matters: Moves open models closer to “product-grade” alignment and safety, enabling replication and extension.

## 5. Experimental Analysis
- Evaluation methodology
  - Base models: Aggregated benchmarks across code, commonsense, world knowledge, reading, math, and popular composites (MMLU/BBH/AGIEval) with standardized shot settings (Table 3; Section 2.3).
  - Chat helpfulness: ~4,000 prompts (single and multi‑turn), 3 independent raters per comparison on a 7‑point scale; win/tie/loss vs. open and closed baselines (Section 3.4.2; Table 32). Inter‑rater reliability (Gwet’s AC2) ranges 0.37–0.55 depending on model pairing.
  - Chat safety: ~2,000 adversarial prompts across three risk categories; 3 raters; violation if mean rating ≤2 on a 5‑point scale (Section 4.4).
  - Automatic safety: Truthfulness (TruthfulQA), toxicity (ToxiGen), sentiment bias (BOLD) (Sections 4.1, 4.4; Tables 11, 14, 46–50).

- Main quantitative results
  - Base model strength
    - `Llama 2 70B` improves over LLaMA‑1 65B by ~5 pts on MMLU and ~8 pts on BBH (Table 3). It is close to GPT‑3.5 on MMLU (68.9 vs. 70.0) and GSM8K (56.8 vs. 57.1), but lags on HumanEval code (29.9 vs. 48.1) (Table 4).
  - Helpfulness human evaluation
    - > “Llama 2-Chat 70B has a win rate of 36% and a tie rate of 31.5% relative to ChatGPT” (Figure 12). It also substantially outperforms open-source chat baselines; e.g., `Llama 2-Chat 34B` has >75% win rate vs. Vicuna‑33B and Falcon‑40B on the paper’s prompt set (Figure 12).
    - System prompts matter: Removing a system prompt from ChatGPT increases `Llama 2-Chat 70B`’s win rate from 36% to 44% (Figure 30 left).
  - Safety human evaluation
    - `Llama 2-Chat` exhibits low violation rates across sizes, competitive with or lower than ChatGPT and lower than Vicuna/MPT (Figure 17a), while preserving higher helpfulness ratings than more terse but “safe” baselines like Falcon (Figure 17b).
    - Multi‑turn prompts are harder for every model, but `Llama 2-Chat` degrades less than others (Figure 18).
    - Category breakdown shows slightly higher violations on “unqualified advice,” often due to missing disclaimers (Figure 19).
  - Automatic safety
    - Toxicity: After tuning, `Llama 2-Chat` toxicity is effectively ~0% on ToxiGen across sizes (Table 14), a major drop from base models (e.g., Llama 2 70B from 24.60 to 0.01; Tables 11 and 14).
    - Truthfulness: `Llama 2-Chat 70B` improves from 50.18 → 64.14 “truthful and informative” (Table 14); ChatGPT is higher (78.46).
    - Sentiment bias: Fine‑tuning increases positive sentiment scores across many BOLD demographic groups compared to base models (Tables 46–50).
  - Reward model validation
    - Helpfulness and Safety RMs outperform OpenAssistant and even GPT‑4 on internal tests (Table 7); accuracy grows with data/model scale (Figure 6).

- Ablations and robustness checks
  - Context length 4k improves long‑context tasks (SCROLLS; Table 16) with stable general performance (Table 17).
  - `GQA` vs. `MHA/MQA`: GQA quality close to MHA, better than MQA, with significant throughput gains at larger batches (Table 18; Figure 24).
  - Ranking‑loss margin: Improves accuracy on clear pairs; shifts RM score distribution, suggesting PPO reward calibration is important (Table 28; Figure 27).
  - Safety auxiliary loss in RM: Improves unsafe recall from 73.0% → 90.4% (Table 29).
  - Safety data scaling: Safety scores improve monotonically; helpfulness remains near constant; false refusals rise mainly on a “borderline” set with sensitive words (Figure 15; Figure 33; Table 41).
  - Contamination analysis: Only HellaSwag and MMLU‑Humanities show evidence of contamination benefits by the stringent token‑skipgram method; overall MMLU effect is modest (Table 51).

- Are claims supported?
  - Yes—`Llama 2-Chat` surpasses open chat models broadly (Figure 12) and demonstrates state‑of‑the‑art open‑model safety on ToxiGen (Table 14) with transparent procedures. At the same time, closed models (e.g., GPT‑4) retain a lead on several academic benchmarks (Table 4; TruthfulQA in Table 14). The paper acknowledges helpfulness/safety evaluation subjectivity and prompt coverage limits (Figures 1–3, 12; Sections 3.4.2, 4.4).

- Failure cases and trade‑offs
  - Early RLHF rounds showed forgetting (loss of rhyme ability), mitigated by retaining top samples from prior rounds (Section 3.2.3).
  - Safety tuning introduces false refusals on sensitive-but-benign prompts (Figure 33; examples in Table 41).
  - Falcon’s lower violation rate on single‑turn safety often comes with lower helpfulness (short, non‑committal answers) (Figure 17b discussion).

## 6. Limitations and Trade-offs
- Data and language coverage
  - Pretraining is predominantly English (≈90% by documents), with limited multilingual data (Table 10). The model’s performance beyond English is “fragile” (Section 5.2).

- Scope of evaluations
  - Helpfulness set excludes coding/reasoning prompts (Section 3.4.2 Limitations), so results do not reflect these domains.
  - Human evaluations are subjective; IRR for helpfulness is moderate (0.37–0.55 AC2), higher for safety (up to 0.95 across batches; Section 4.4).

- Safety–helpfulness tension
  - Even with separate RMs and scaling showing stable average helpfulness (Figure 15), safety scaling increases false refusals on the borderline set (Figure 33). Some benign prompts with sensitive words are refused (Table 41).

- Compute and deployment constraints
  - Training costs are high (3.3M GPU hours; Table 2). Inference for very large models remains resource‑intensive, though `GQA` helps (Figure 24).
  - PPO required engineering workarounds for generation speed under FSDP (Section 3.2.3).

- Data contamination
  - Some benchmarks likely benefited from data overlap (HellaSwag, MMLU‑Humanities; Table 51). Results are still informative but should be interpreted with this caveat.

- Release scope
  - The 34B model was withheld initially due to insufficient red teaming time (Section 1 footnote).

- Open questions
  - Long‑term safety under adversarial distribution shift; multilingual alignment; persistent memory beyond 4k tokens; tool‑use governance.

## 7. Implications and Future Directions
- Field impact
  - Provides a practical, end‑to‑end open recipe for building assistant‑grade LLMs with documented choices and ablations. This lowers the barrier for research and responsible commercialization and creates a baseline for safety and alignment research.

- Follow‑up research enabled
  - Reward modeling
    - Better cross‑domain/general RMs; exploring multi‑objective RMs and Pareto frontiers between helpfulness, safety, and other values.
    - Calibration-aware rewards and PPO techniques robust to reward distribution shifts (see whitening and margin effects in Section 3.2.2; Figure 27).
  - Safety
    - Improved borderline detection to reduce false refusals without increasing violations (Figures 33, 19).
    - Multilingual safety data, non‑English red teaming, and culturally aware guidelines (Table 10 shows language skew).
    - Stronger defenses against jailbreaks using targeted distillation and adversarial training.
  - Long‑context and memory
    - Extend `GAtt` to dynamic system messages and >4k contexts, and study how it interacts with retrieval and memory modules (Section 3.3; Appendix A.3.5 notes promise beyond 2048 tokens on LLaMA‑1).
  - Tool use and planning
    - Formalize the observed zero‑shot tool chaining (Figure 23; Table 15) and evaluate safety of tool affordances.
  - Evaluation science
    - More comprehensive, longitudinal, and task‑oriented evaluations (e.g., multi‑turn tasks scored end‑to‑end; Section 3.4.2 limitations).
    - Continued contamination auditing (Appendix A.6).

- Practical applications
  - Safer assistant chatbots for customer service, productivity, and education.
  - Domain- and policy‑adaptable safety: the modular pipeline (separate RMs, targeted distillation) supports sector‑specific safety standards (finance, health, legal).
  - Platform features like role‑play or “style” adherence across turns via `GAtt`, improving UX consistency.

> In summary, Llama 2 contributes (i) stronger open base models (Table 3), (ii) a transparent, scalable alignment pipeline with separate helpfulness and safety reward models, (iii) targeted safety methods that substantively reduce toxic and unsafe outputs (Table 14; Figures 17–19), and (iv) a simple but effective method to maintain system instructions across long, multi‑turn dialogues (Figure 10; Table 30). While closed models still hold advantages on certain benchmarks (Table 4; TruthfulQA in Table 14), this work substantially raises the floor for open, responsibly aligned assistants and supplies the community with a reproducible blueprint to improve them further.
