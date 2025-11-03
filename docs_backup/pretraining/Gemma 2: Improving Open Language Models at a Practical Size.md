# Gemma 2: Improving Open Language Models at a Practical Size

**ArXiv:** [2408.00118](https://arxiv.org/abs/2408.00118)
**Authors:** Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk
**Institutions:** Google DeepMind

## 🎯 Pitch

Gemma 2 presents a breakthrough in small-to-mid-size language models by leveraging an innovative approach where knowledge distillation replaces traditional next-token training. This technique, coupled with architectural optimizations, significantly enhances model efficiency and performance, enabling smaller models to rival far larger counterparts, thus making them ideal for resource-constrained applications without compromising quality.

---

## 1. Executive Summary
Gemma 2 introduces three open, decoder‑only language models (`2B`, `9B`, `27B` parameters) that pair architectural tweaks with an aggressive use of knowledge distillation to push small‑to‑mid‑size models to near–state‑of‑the‑art quality. The standout idea is to replace standard next‑token training with distillation from a stronger “teacher” model and to train far beyond compute‑optimal token counts, yielding models that outperform peers of the same size and rival models 2–3× larger on many benchmarks.

## 2. Context and Motivation
- Problem addressed
  - Small open models have improved mostly by more training data (longer training), but such gains scale poorly. The paper notes that recent small models consume “up to 15T tokens to improve the state of the art by less than 1–2%” (Section 1, citing AI@Meta 2024). This suggests small models are under‑trained in an information sense, not just a token‑count sense.
- Why it matters
  - Practical deployment often needs compact models (resource‑constrained devices, lower inference cost). If small models can be trained to “think” more effectively without brute‑force scaling, they could deliver high quality at practical sizes.
- Prior approaches and gaps
  - Prior small models largely extended token budgets and adopted standard objectives (next‑token prediction). Scaling laws (Hoffmann et al., 2022) predict diminishing returns from just adding tokens.
  - Knowledge distillation exists but is often used to speed up training rather than as the primary long‑horizon training objective.
- Positioning of this work
  - Gemma 2 reframes distillation as the main objective for small models and couples it with architectural choices aimed at throughput and stability. The paper trains the `2B` and `9B` models “on a quantity of tokens that is more than 50× the compute‑optimal quantity” (Section 1), using a teacher’s full probability distribution to provide richer learning signals per token.

## 3. Technical Approach
This section explains the architecture, training pipeline, and post‑training recipe.

- Model architecture (Section 2; Table 1, Table 2)
  - Base: decoder‑only Transformer with `8192` token context, RoPE positional encoding, and `GeGLU` activations.
  - Interleaved local–global attention:
    - A “local sliding window” attention layer alternates with a “global attention” layer every other layer.
    - Local layers attend over a sliding window of `4096` tokens; global layers attend over the full `8192` context (Section 2: “We alternate between a local sliding window attention … and global attention … The sliding window size … is set to 4096 tokens, while the span of the global attention layers is set to 8192 tokens.”).
    - Why: local attention reduces compute, while periodic global layers restore long‑range information flow.
  - `GQA` (Grouped‑Query Attention) with `num_groups = 2`
    - Definition: in GQA, multiple attention heads share the same Key/Value projections (grouped K/V), reducing memory and speed costs while keeping separate Query projections for expressivity.
    - Chosen due to “increased speed at inference time while maintaining downstream performance” (Section 2). Table 8 shows near‑parity between standard Multi‑Head Attention (`MHA`) and `GQA`, favoring GQA for efficiency.
  - Normalization and stability
    - `RMSNorm` is used both pre‑norm and post‑norm for attention and feed‑forward blocks (Section 2). Pre‑norm stabilizes gradients in deep nets; post‑norm helps stabilize outputs.
  - Logit soft‑capping
    - To prevent excessively large logits (which can destabilize training and inference), they cap logits using `logits ← soft_cap * tanh(logits / soft_cap)`, with `soft_cap = 50.0` for self‑attention layers and `30.0` for the final layer (Section 2).
  - Sizes and layouts (Table 1)
    - `2B`: 26 layers, `d_model=2304`, `8` heads (`4` KV heads), `head_size=256`, `FFN dim=18432`.
    - `9B`: 42 layers, `d_model=3584`, `16` heads (`8` KV), `head_size=256`, `FFN dim=28672`.
    - `27B`: 46 layers, `d_model=4608`, `32` heads (`16` KV), `head_size=128`, `FFN dim=73728`.
    - Large shared vocabulary (`256,128` entries) increases embedding parameters (Table 2).
- Pre‑training data and objective (Section 3)
  - Token budgets: `27B` trained on `13T` tokens (from scratch), `9B` on `8T`, `2B` on `2T` (Section 3.1).
  - Data mixture: primarily English, drawn from web, code, and science sources; filtered to remove unsafe content and to decontaminate evaluation sets (Section 3.1).
  - Tokenizer: SentencePiece with byte‑level encodings and digit splitting, `256k` vocab (Section 3.1).
- Knowledge distillation as the main objective for `2B` and `9B` (Section 3.2)
  - Definition: train a smaller “student” to match a larger “teacher” model’s probability distribution over the next token, not just the one‑hot target. Formally:
    > minimize over PS:  Σx [ − PT(x | xc) · log PS(x | xc) ]  (Section 3.2)
  - Intuition: the teacher’s full distribution conveys “dark knowledge” (relative probabilities among plausible tokens), delivering richer gradients, especially helpful when training on massive token counts beyond compute‑optimal limits.
- Training infrastructure and software (Section 3.3; Table 3)
  - Hardware: TPUv5e (`2B`), TPUv4 (`9B`), TPUv5p (`27B`), scaling up to `6144` chips for `27B`.
  - Parallelism: data replication, model sharding, ZeRO‑3‑like optimizer sharding, Pathways for cross‑pod reduction, `JAX` single‑controller programming, `GSPMD` partitioner, MegaScale XLA compiler.
  - Carbon footprint: estimated `1247.61 tCO2eq`, with data centers operating under Google’s carbon‑neutral policy (Section 3.4).
- Post‑training for instruction tuning (Section 4; Table 4, Table 5)
  - SFT (supervised fine‑tuning): on a mix of human and synthetic prompt–response data, heavily leveraging teacher‑generated responses; also distillation on the student’s distribution during SFT (Section 4).
  - RLHF (reinforcement learning from human feedback): reward model “an order of magnitude larger than the policy,” oriented toward multi‑turn conversation (Section 4).
  - Model merging: weight‑space averaging of multiple runs to improve overall performance (Section 4).
  - Safety‑aware data filtering and formatting: standardized control tokens for multi‑turn chat; updated schema ends model outputs with `<end_of_turn><eos>` (Table 4, Table 5).

Analogy for the core idea: Instead of teaching a student only the single correct answer per question (next‑token), the student watches the teacher’s full answer key showing partial credit for close answers (full probability distribution). Practiced across far more “questions” than usual, the student learns deeper patterns with similar study time.

## 4. Key Insights and Innovations
- Distillation as a long‑horizon training objective for small models
  - Novelty: treat knowledge distillation not just as a compression or speed‑up technique but as the primary training objective for small models, over very large token counts.
  - Evidence: 
    > Table 6: `2B` trained 500B tokens “from scratch” vs “distilled” shows “Average (3 bench.) 60.3 → 67.7”.  
    > Table 7: Distillation lowers perplexity across 200M, 400M, and 1B models (e.g., at 1B: `from scratch 17` vs `distilled 15`).
  - Significance: This reframing helps small models close the gap to much larger models without prohibitive parameter growth.
- Interleaved local–global attention for long contexts with efficiency
  - Difference: alternates local sliding window (`4096`) with full `8192` global attention per layer (Section 2).
  - Why it matters: retains long‑range modeling while reducing attention compute on many layers. Also enables an inference‑time speed/quality trade‑off (Table 10 shows minimal perplexity change when shrinking the local window from 4096 to 1024).
- Adoption of `GQA` to cut inference cost with negligible quality loss
  - Evidence: 
    > Table 8: `MHA 50.3` vs `GQA 50.8` (average across 4 benches).  
  - Benefit: memory and speed benefits of grouped K/V outweigh tiny differences in average scores.
- Deeper vs wider preference at fixed parameter budget
  - Evidence:
    > Table 9: “Wide 50.8” vs “Deep 52.0” (average across 4 benches).
  - Insight: additional depth can be more useful than width for 9B‑scale models under these training regimes.
- Format robustness tracking
  - Observation: sensitivity to prompting/evaluation format measured by std‑dev on MMLU across 12 formats (Table 11). 
    > `Gemma 2 2B`: 2.1; `Gemma 2 9B`: 0.9; `Gemma 2 27B`: 1.0; `Mistral 7B`: 6.9.
  - Significance: lower variance suggests more stable performance under reasonable formatting variations.

## 5. Experimental Analysis
- Evaluation methodology
  - Pre‑training quality: standard academic benchmarks such as MMLU, GSM8K, ARC‑c, HellaSwag, Winogrande (Table 12). 
  - Comparative baselines: models of similar or larger sizes (e.g., Qwen1.5 32B, LLaMA‑3 70B; Table 12).
  - Post‑training quality: human preference studies, Chatbot Arena Elo (Table 14), instruction following and safety SxS vs GPT‑4o (Table 15), multi‑turn conversations (Table 16), and few‑shot performance changes from IT (Table 17).
  - Ablations: distillation vs scratch (Table 6–7), `GQA` vs `MHA` (Table 8), deep vs wide (Table 9), sliding window change at inference (Table 10), format robustness (Table 11).
  - Safety and memorization: toxicity/bias/factuality suites (Table 18), memorization analysis (Figure 1), and assurance studies (offensive cybersecurity, code vulnerabilities, self‑proliferation, persuasion; Tables 19–25).
- Main quantitative results
  - Distillation gains for small models
    > Table 6: `2B` @500B tokens: average across 3 benchmarks improves from `60.3` (scratch) to `67.7` (distilled).  
    > Table 7: perplexity reductions across model sizes with a 7B teacher (e.g., 1B: `17` → `15`).
  - Pre‑trained 27B vs larger baselines (Table 12)
    > `Gemma‑2 27B` MMLU `75.2`, GSM8K `74.0`, ARC‑c `71.4`, HellaSwag `86.4`, Winogrande `83.7`.  
    > Outperforms Qwen1.5 32B on most metrics, and is “only a few percent below LLaMA‑3 70B despite being 2.5× smaller and trained on 2/3rds less data.”
  - Pre‑trained `2B`/`9B` vs prior open models (Table 13)
    - Average across all benchmarks:  
      > `Gemma‑1 2B 44.2` → `Gemma‑2 2B 48.7` (+4.5 points);  
      > `Gemma‑1 7B 57.9` → `Gemma‑2 9B 64.9` (+7.0);  
      > `Gemma‑2 27B 69.4`.
    - Per‑task highlights (Gemma‑2 9B): MMLU `71.3`, GSM8K `68.6`, BBH `68.2`, MBPP `52.4`.
  - Post‑training human preference: Chatbot Arena Elo (Table 14)
    > `gemma‑2‑27b‑it`: Elo `1218`, ranked above `llama‑3‑70b‑instruct 1206`.  
    > `gemma‑2‑9b‑it`: Elo `1187`, comparable to `gpt‑4‑0314 1186`.  
    > `gemma‑2‑2b‑it`: Elo `1126`, above `gpt‑3.5‑turbo‑0613 1116`.
  - Instruction following & safety (Table 15)
    > Instruction following (single‑sided): `Gemma‑2 9B 34.1% ± 3.0%` (vs `Gemma‑1.1 7B 24.3% ± 1.9%`).  
    > Safety (Win/Tie/Loss vs GPT‑4o): `Gemma‑2 9B 48.2% / 19.2% / 28.3%`; `Gemma‑2 2B 53% / 9% / 38%`.
  - Multi‑turn conversations (Table 16; 500 scenarios; 1–5 scale)
    > User satisfaction: `Gemma‑1.1 7B 3.32` → `Gemma‑2 27B 4.20`.  
    > Goal achievement: `3.36` → `4.24`.
  - IT vs PT on few‑shot (Table 17)
    > MMLU (2B/9B/27B): `52.2→56.1`, `71.3→72.3`, `75.2→76.2`.  
    > MBPP: `30.2→36.6`, `52.4→59.2`, `62.6→67.4`.
  - Memorization (Figure 1)
    > “Significantly lower memorization rates across‑the‑board,” with exact memorization below `0.1%` overall. Approximate memorization increases are small relative to prior models.
    > Personal data analysis: “no instances of high‑severity,” and only `0.00026%` of memorized data contained lower‑severity PII (Section 7).
  - Safety benchmarks (Table 18)
    - The table lists RealToxicity, CrowS‑Pairs, BBQ, Winogender/Winobias, TruthfulQA, etc. Trends vary by metric; larger Gemma 2 models often match or improve over Gemma 1.1, with some metrics favoring specific sizes. The table notes which direction (higher/lower) is better when bolded.
  - Assurance results (Tables 19–21)
    - Offensive cybersecurity (CTFs; Table 19): `Gemma‑2 27B` solves `34/76` InterCode tasks (45% success on that subset), `1/13` internal CTF, `0/13` Hack the Box; well below `Gemini 1.5 Pro 62/76` but above older open models (e.g., CodeGemma 7B).
    - Code vulnerability detection (Table 20): near‑chance accuracy on several datasets; roughly on par with Gemini 1.5 Pro on some suites (e.g., SecretPatch `72%` vs `67%`).
    - Self‑proliferation (Table 21): `0/10` end‑to‑end tasks; `1/10` “all milestones” scenarios; `22/45` milestones overall; indicates limited autonomous capability without human intervention.
  - Persuasion studies (Tables 22–25)
    - Rapport (“Charm Offensive”, Table 22): high perceived traits (e.g., `Personal connection 80%`, `Trustworthy 87%` at 27B), comparable to strong proprietary models.
    - Hidden agenda (Table 23): persuades a fraction of participants to click links/find info/run code (`34%/9%/11%`), within the range of Gemini models.
    - Donations (“Money Talks”, Table 24): no significant increase vs baseline.
    - “Web of Lies” (Table 25): shifts beliefs toward correct facts (`18% ± 5%`) but minimal shift toward incorrect (`1% ± 4%`), weaker than human confederates at inducing false beliefs.
- Do the experiments support the claims?
  - Yes for core claims:
    - Distillation increases small‑model quality (Tables 6–7).
    - The `27B` pre‑trained model competes with larger baselines (Table 12).
    - Instruction‑tuned Gemma 2 models achieve top open‑weights standings on Chatbot Arena (Table 14).
  - Robustness and ablations are present (Tables 6–11), and assurance/safety sections probe potential harms and capabilities (Tables 18–25, Figure 1).
- Nuances and trade‑offs seen in results
  - Format sensitivity improves with size but the `2B` model is less robust (Table 11).
  - Safety benchmarks show mixed patterns across datasets (Table 18), suggesting gains depend on the metric and domain.
  - Assurance results show capability increases on some cyber tasks but remain far below cutting‑edge proprietary systems and fail end‑to‑end autonomy tests (Tables 19–21).

## 6. Limitations and Trade-offs
- Reliance on a high‑quality teacher
  - Distillation quality depends on the teacher. Any biases or errors in the teacher distribution may be inherited. The paper does not detail the exact teacher identity for the final models, limiting replicability of the precise effect size.
- Training compute and data
  - Although parameter counts are small, token budgets are very large (e.g., `8T` for `9B`, `2T` for `2B`, `13T` for `27B`; Section 3.1). The approach shifts cost from parameters to data/compute for long training runs. Carbon impact is reported (`1247.61 tCO2eq`; Section 3.4), but the training remains resource‑intensive.
- Limited multilingual and multimodal scope
  - The models are “not trained specifically for state‑of‑the‑art multilingual capabilities” and are text‑only (Section 3.1). Embedding size is large due to the `256k` multilingual‑friendly vocabulary (Table 2), which increases memory footprint without providing full multilingual or multimodal coverage.
- Formatting sensitivity at small scale
  - The `2B` model shows higher variance to formatting on MMLU (Table 11), which can affect reliability in real applications if prompts vary.
- Safety strengths but residual risks
  - Assurance tests show persuasion and some offensive cybersecurity capabilities (Tables 22–23, 19), even if far from frontier models. Developers still need system‑level safeguards.
- Inference trade‑offs
  - Reducing sliding window size can speed inference but slightly raises perplexity (Table 10), implying a quality–latency trade‑off that downstream users must tune.

## 7. Implications and Future Directions
- How this work shifts the landscape
  - It validates a path to high‑quality small models by maximizing information per token via distillation, rather than only scaling parameters or raw token counts. This changes the optimization target for practical models: invest in a better training signal (teacher distributions) over longer runs.
- Follow‑up research
  - Distillation design: choice of teacher(s), temperature scaling, curriculum schedules, and domain‑specific teachers for code, math, or reasoning.
  - Architecture: further exploration of local–global layer patterns, dynamic window sizes, or learned schedules; combining with Mixture‑of‑Experts while retaining small active parameter counts.
  - Data and safety: better methods to measure and reduce approximate memorization; more transparent reporting of data mixtures; domain‑targeted safety RLHF to reduce persuasion on harmful tasks without hurting helpfulness.
  - Robustness: reduce formatting sensitivity for very small models; evaluate cross‑lingual generalization given the large vocabulary and partially multilingual data.
- Practical applications
  - Edge and on‑prem deployment: `2B` and `9B` models with strong Chatbot Arena rankings (Table 14) are attractive for private or resource‑constrained settings.
  - Multi‑turn assistants: improved satisfaction and goal achievement (Table 16) suggest suitability for customer support, tutoring, and planning tools, with attention to safety guardrails.
  - Coding and reasoning helpers: solid gains on MBPP, HumanEval, GSM8K, and BBH (Table 13) make these models broadly useful as coding/math copilots, though further fine‑tuning or tool integration may be required for high‑stakes tasks.

Overall, Gemma 2 demonstrates that re‑thinking the training objective—feeding small models dense, distributional supervision from a strong teacher over very long token budgets—can deliver outsized quality at practical parameter counts. The comprehensive evaluations, ablations, and safety analyses ground the claim that such models can be both capable and responsibly deployable, while highlighting remaining gaps in multilinguality, autonomy, and robustness that future work can address.
