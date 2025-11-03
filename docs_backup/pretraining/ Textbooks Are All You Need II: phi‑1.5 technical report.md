# Textbooks Are All You Need II: phi‑1.5 technical report

**ArXiv:** [2309.05463](https://arxiv.org/abs/2309.05463)
**Authors:** Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, Yin Tat Lee
**Institutions:** Microsoft Research

## 🎯 Pitch

The `phi-1.5` model demonstrates that high-quality, synthetic 'textbook-like' data can rival traditional, scale-heavy language models, achieving remarkable results in reasoning and language tasks with just 1.3B parameters. This innovation significantly reduces computational demands and toxicity risks, challenging the prevailing belief that massive datasets are essential, and opening new avenues for developing more efficient, safe, and versatile AI systems.

---

## 1. Executive Summary
This technical report introduces `phi-1.5`, a 1.3B-parameter language model trained primarily on LLM-generated “textbook-like” synthetic data, and shows that careful data curation can substitute for massive scale. With only 150B training tokens and modest compute (Table 1), `phi-1.5` matches or outperforms much larger open models (5–10× bigger) on several common-sense, language understanding, and especially multi-step reasoning tasks (Tables 2–4), while also exhibiting lower toxicity tendencies than web-trained peers (Figure 2, Section 4).

## 2. Context and Motivation
- Problem addressed:
  - Can small language models achieve strong reasoning and language abilities without trillion-scale parameters/tokens?
  - How far can we push “data quality over data quantity,” especially for common-sense reasoning and basic coding?

- Why it matters:
  - Practical: Training and serving smaller models is cheaper, faster, and greener (Table 1 shows <3 ms/token inference and 3.5 GB memory at 2k context for `phi-1.5` vs ~14 ms and 18 GB for a 7B baseline).
  - Scientific: Is scale indispensable, or can curated training data induce advanced behaviors (e.g., step-by-step reasoning, in-context learning) at small scale?
  - Responsible AI: Web-trained models inherit internet toxicity and bias; synthetic “textbook-like” data might reduce these risks (Section 4, Figure 2).

- Prior approaches and gaps:
  - Scaling: Frontier models have hundreds of billions of parameters and near-trillion-token corpora (Intro cites PaLM 540B parameters and 780B tokens).
  - Small models: Prior “TinyStories” (10M params) and `phi-1` (1.3B) showed coherent English and near-SOTA Python coding using synthetic “textbook” data (Intro; Section 2).
  - Gap: Common-sense reasoning and general language understanding at small scale remained underexplored, and it wasn’t clear how much web data helps vs synthetic data.

- Positioning:
  - Extends the “Textbooks Are All You Need” line beyond code to natural-language common sense and reasoning (Section 1).
  - Compares three variants to isolate data effects (Section 2.4):
    - `phi-1.5` (mostly synthetic, no web)
    - `phi-1.5-web-only` (filtered web only)
    - `phi-1.5-web` (mixture of synthetic + filtered web + code)

## 3. Technical Approach
This is an empirical study that controls model size and training recipe while varying data sources and quality.

- Model architecture (Section 2.1):
  - 24-layer Transformer, 32 attention heads with head dimension 64.
  - Rotary position embeddings (RoPE) with rotary dimension 32. RoPE encodes token positions by rotating key/query vectors; it preserves relative positions and benefits extrapolation.
  - Context length 2048 tokens; tokenizer from `codegen-mono`.
  - Uses FlashAttention (memory-/IO-efficient exact attention) to accelerate training.

- Training data (Section 2.2):
  - Core idea: Train on “textbook-like” synthetic data—LLM-generated, instruction-style educational text with explanations, exercises, and answers—rather than raw web text.
  - Construction:
    - Curate ~20,000 topics to seed generation (e.g., science concepts, daily activities, theory of mind).
    - Use prompts (sometimes seeded with web samples for diversity) to generate ~20B tokens of synthetic educational content.
    - Combine with `phi-1`’s 7B training tokens; the only non-synthetic portion here is ~6B tokens of filtered code used previously for `phi-1`.
  - Mixture during training:
    - `phi-1.5`: 80% new synthetic NLP data + 20% `phi-1` data (Section 2.3).
    - `phi-1.5-web-only`: 95B tokens of filtered web + code (Section 2.4).
    - `phi-1.5-web`: ~40% filtered web + 40% synthetic NLP + 20% code (Section 2.4).

  - What “filtered web” means:
    - Start from Falcon RefinedWeb (88B tokens) plus code from The Stack and StackOverflow (7B tokens).
    - Apply filtering as in the earlier `phi-1` work to remove low-quality or unsafe content (Section 2.4 and [GZA+23] cited there).

- Optimization details (Section 2.3):
  - Train from random initialization for 150B tokens.
  - Adam with betas (0.9, 0.98), epsilon 1e-7, weight decay 0.1.
  - Constant learning rate 2e-4 with no warmup.
  - Batch size 2048, fp16, DeepSpeed ZeRO Stage 2 for memory efficiency.

- Compute and deployment profile (Table 1):
  - `phi-1.5`: ~1.5K GPU-hours on a single A100-80G equivalent, inference <3 ms/token, ~3.5 GB memory at 2048 context.
  - `phi-1.5-web`: ~3K GPU-hours, same inference/memory characteristics.
  - For perspective, a 7B model (Llama-7B line in Table 1) is reported at >80K GPU-hours and ~14 ms/token, ~18 GB memory.

- No instruction tuning or RLHF (Section 2.4, Remark):
  - All models are base (completion) models; they’ve not been aligned to follow instructions or refuse unsafe prompts.
  - Despite that, “exercise/answer” patterns in the synthetic data often teach basic instruction following and simple chat ability (Section 5).

- Why this design:
  - Hold architecture and size fixed to isolate the effect of training data quality/mixture (synthetic vs filtered web).
  - Emphasize a clean, simple training regime to highlight data’s contribution (Section 2.3 note).

- How results are obtained:
  - Use a consistent in-house evaluation pipeline across models (Figure 1 note), largely via the LM-Eval Harness, to avoid cross-benchmark inconsistencies (Section 3).

## 4. Key Insights and Innovations
- Data quality can substitute for scale (fundamental innovation):
  - With only 1.3B parameters and 150B tokens, `phi-1.5` achieves performance comparable to or better than 5–10× larger models on many tasks (Figure 1; Tables 2–4). This challenges the prevailing “scale-is-all-you-need” narrative by isolating the role of curated synthetic data.

- Synthetic “textbook-like” data induces reasoning behaviors at small scale (fundamental innovation):
  - The model demonstrates “think step by step” abilities and rudimentary in-context learning without instruction tuning (Section 5 examples). The training data’s exercise/answer format seems to teach procedural reasoning skills.

- Clear decomposition of data effects (methodological insight):
  - By training three variants—`phi-1.5` (synthetic-heavy), `phi-1.5-web-only` (web-only), and `phi-1.5-web` (mixture)—the report shows:
    - Web-only significantly improves over prior 1–3B models but underperforms the synthetic-heavy or mixed models on many benchmarks (Tables 2–4).
    - Mixing filtered web with synthetic further boosts multi-step reasoning and coding (Table 4), suggesting complementary strengths.

- Lower toxicity tendencies without alignment (practical/safety insight):
  - Despite being base models, `phi-1.5` variants show improved safety scores relative to web-trained baselines on ToxiGen and targeted adversarial prompts (Section 4; Figure 2), plausibly because they avoid direct exposure to toxic web text.

## 5. Experimental Analysis
- Evaluation setup (Section 3; Figure 1; Tables 2–4):
  - Benchmarks span:
    - Common-sense reasoning: WinoGrande, ARC-Easy, ARC-Challenge, BoolQ, SIQA.
    - Language understanding/knowledge: PIQA, HellaSwag, OpenBookQA, SQuAD (Exact Match), MMLU (2-shot).
    - Multi-step reasoning: GSM8K (grade-school math), HumanEval and MBPP (entry-level Python coding). “Zero-shot pass@1” is used for coding; for GSM8K, the prompt strategy includes “via coding,” i.e., asking the model to write small programs to compute answers.
  - Metrics:
    - Accuracy for choice-style tasks; Exact Match (SQuAD); pass@1 for code tasks; zero-shot unless specified (e.g., MMLU 2-shot).
  - Baselines:
    - Open models from 1.3B to 13B+ parameters (OPT-1.3B, GPT-Neo-2.7B, GPT2-XL-1.5B, Falcon-rw-1.3B/7B, MPT-7B, Llama-7B, Llama 2-7B, Vicuna-13B, and Llama-65B for reasoning/coding reference).

- Headline results
  - Common-sense reasoning (Table 2):
    - > “`phi-1.5 (1.3B)`: WinoGrande 0.734, ARC-Easy 0.756, ARC-Challenge 0.444, BoolQ 0.758, SIQA 0.526.”
    - Comparable to or slightly better than `Llama 2-7B` on most of these (e.g., WinoGrande 0.691, ARC-Challenge 0.434), and close to `Vicuna-13B` (e.g., WinoGrande 0.708).
    - `phi-1.5-web` is similar or marginally stronger than `phi-1.5` (e.g., WinoGrande 0.740, ARC-Challenge 0.449).
    - `phi-1.5-web-only` (web-only) beats other ~1–3B models but trails the synthetic-heavy variants.
  - Language understanding/knowledge (Table 3):
    - Mixed picture:
      - `phi-1.5`: PIQA 0.766 vs Llama 2-7B 0.781 (slightly lower), HellaSwag 0.476 vs 0.571 (lower), OpenBookQA 0.372 vs 0.314 (higher), SQuAD EM 0.72 vs 0.67 (higher), MMLU 0.376 vs 0.453 (lower).
      - `phi-1.5-web` is similar on most and stronger on SQuAD (0.74 EM) but still below Llama 2-7B on MMLU (0.379 vs 0.453).
    - Interpretation: synthetic data confers strong generalization to some QA-style tasks (SQuAD, OpenBookQA), but broad academic knowledge (MMLU) still benefits from more diverse or richer corpora.
  - Multi-step reasoning and coding (Table 4):
    - > “`phi-1.5`: GSM8K 40.2 (via coding), HumanEval 34.1, MBPP 37.7.”
    - > “`phi-1.5-web`: GSM8K 44.6 (via coding), HumanEval 41.4, MBPP 43.5.”
    - These exceed all listed 7B models and even `Llama-65B` on some coding metrics (e.g., `phi-1.5-web` 41.4 vs `Llama-65B` 23.7 on HumanEval; MBPP 43.5 vs 37.7). On GSM8K, `phi-1.5-web` reaches 44.6 vs `Llama-65B` 50.9—close given the 50× parameter gap.
    - Web data helps most here: the mixture (`phi-1.5-web`) outperforming synthetic-only `phi-1.5`, and both beating web-only at 1.3B.

- Safety and toxicity (Section 4; Figure 2):
  - > “Scores range 0–1; higher is safer.” `phi-1.5` and `phi-1.5-web` generally score above 1–3B web-trained baselines across 13 demographic groups.
  - A manual stress test with 86 adversarial prompts: `phi-1.5` passes 47, fails 34, and “did not understand” 4; by comparison, `Llama2-7B` fails 54 and “did not understand” 13 (Section 4). This is notable given `phi-1.5` has no alignment fine-tuning.

- Qualitative capabilities (Section 5):
  - Despite being base models, the synthetic training produces:
    - Direct completion with scenario consistency.
    - Chain-of-thought when prompted “Let’s think step by step.”
    - Basic instruction following and chat-like behavior (“Person A:… Person B:…”).
    - Usable—but imperfect—Python generation and code explanation.
  - The paper shows both successful generations and small mistakes (e.g., the socket example notes inaccuracies in explanation), reinforcing that the models are capable but not foolproof.

- Do the experiments support the claims?
  - Yes for the central claims:
    - Small model + high-quality synthetic data can rival much larger models on common-sense reasoning and beat them on coding and math reasoning (Tables 2–4).
    - Mixed or synthetic data reduces toxicity relative to web-only training (Figure 2; manual probe set).
  - Caveats:
    - Results come from the authors’ standardized evaluation pipeline and may differ slightly from external reports (Figure 1 note).
    - MMLU remains lower than Llama 2-7B, indicating limits in broad factual/academic coverage.

- Ablations and robustness:
  - The three variants (`phi-1.5`, `phi-1.5-web-only`, `phi-1.5-web`) act as data ablations showing the contributions of synthetic vs web data (Tables 2–4).
  - Safety evaluation triangulates automated (ToxiGen) and manual prompts (Section 4).

## 6. Limitations and Trade-offs
- Coverage and dependence on teacher LLMs (Section 2.2):
  - Synthetic data is generated from prompts seeded by ~20K topics. Topic selection and prompt engineering are manual and iterative; blind spots in coverage can limit downstream knowledge and skills.
  - The approach implicitly relies on upstream LLMs to generate high-quality “textbooks,” which can encode their own biases or inaccuracies.

- No alignment fine-tuning (Section 2.4 Remark; Section 5):
  - These are base models; they do not refuse unsafe instructions and sometimes continue generations undesirably (e.g., not stopping cleanly). While toxicity is reduced compared to some baselines, it is not eliminated (Section 4).

- Mixed results on broad knowledge (Table 3):
  - MMLU scores lag behind larger or more web-exposed models. The curated/synthetic corpus may not capture the breadth of academic facts, even if it teaches reasoning patterns well.

- Context length and scaling:
  - Context is limited to 2048 tokens. Longer-context tasks or retrieval-heavy workloads are not addressed.

- Compute vs accuracy trade-offs:
  - While compute is dramatically lower than 7B+ models (Table 1), top-tier performance on some tasks (e.g., GSM8K vs `Llama-65B`) still benefits from more parameters and tokens.

- Safety evaluation scope:
  - ToxiGen and the 86-prompt set are informative but not exhaustive. Real-world safety requires broader, adversarially designed probes.

## 7. Implications and Future Directions
- Field-level impact:
  - Demonstrates that careful, concept- and exercise-oriented synthetic corpora can elicit advanced reasoning behaviors in small models. This tilts the research agenda toward “data-centric pretraining” and away from assuming that sheer scale is the only path to capability.

- Practical applications:
  - Edge or on-prem deployments needing low-latency, small-footprint models for reasoning and code synthesis.
  - Educational tools that benefit from step-by-step explanations—even without full instruction tuning.
  - Safer base models for further alignment research due to reduced exposure to toxic web content (Figure 2).

- Research directions:
  - Synthetic data engineering as a discipline:
    - How to systematically pick topics, generate, verify, and diversify textbook modules?
    - Automated coverage analysis to find and fill knowledge gaps (Section 2.2 hints at this becoming a core skill).
  - Hybrid data strategies:
    - Results show web data complements synthetic data for math/coding (Table 4). Explore principled mixing ratios and domain-adaptive filtering.
  - Alignment on top of synthetic pretraining:
    - Apply instruction tuning/RLHF to test how much easier it is to align models that start from “cleaner” priors.
  - Broader benchmarks:
    - Expand to multilingual, long-context, and domain-specific tasks (law, medicine) to assess generality.
  - Interpretability and in-context learning:
    - `phi-1.5` offers a tractable size for mechanistic interpretability and in-context learning studies (Section 6), potentially revealing how textbook patterns translate into internal circuits.

In short, `phi-1.5` makes a strong case that “textbooks are (almost) all you need”—not to replace scale entirely, but to dramatically reduce how much of it is required—especially for reasoning-heavy abilities. The controlled comparisons, concrete compute savings (Table 1), and broad evaluations (Tables 2–4; Figure 2) collectively support the central thesis that data quality and structure are powerful levers for LLM capability.
