# Textbooks Are All You Need II: phi-1.5 technical report

**ArXiv:** [2309.05463](https://arxiv.org/abs/2309.05463)

## 🎯 Pitch

This paper introduces phi-1.5, a 1.3-billion parameter language model trained primarily on high-quality synthetic 'textbook-like' data, achieving common sense and multi-step reasoning performance on par with or better than models five to ten times larger. By demonstrating that carefully constructed synthetic datasets can rival massive web-scale corpora in unlocking advanced language abilities, phi-1.5 paves the way for far more efficient, accessible, and governable AI systems—potentially transforming how the field approaches data and scale in language model training.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces `phi-1.5`, a 1.3B-parameter language model trained primarily on high-quality, synthetic “textbook-like” data that matches or surpasses much larger open-source models on common-sense and multi-step reasoning tasks. Its significance is twofold: it shows that carefully curated synthetic data can substitute massive web-scale corpora for many capabilities, and it delivers strong performance with dramatically lower compute and memory requirements (Table 1).

## 2. Context and Motivation
- Problem/gap addressed:
  - State-of-the-art language models have improved largely by scaling model size and training data (e.g., PaLM’s 540B parameters and 780B tokens). The key question is whether smaller models can achieve comparable capabilities without extreme scale (Introduction, p.1–2).
  - Prior small models typically struggle with commonsense reasoning and multi-step reasoning—tasks that require structured, step-by-step thinking and practical world knowledge.

- Why this matters:
  - Economic and environmental costs of training/deploying giant models are high (Introduction).
  - Smaller, capable models enable broader access, easier experimentation, and better governance.
  - If performance depends more on data quality than sheer size, the development paradigm shifts.

- Prior approaches and their limits:
  - `TinyStories` showed 10M-parameter models can produce coherent English, and `phi-1` (1.3B) achieved near-SOTA Python coding using synthetic “textbook-quality” code/data [EL23, GZA+23].
  - However, whether synthetic “textbook” data can also unlock broader natural-language commonsense reasoning remained open (Introduction, p.1).

- Positioning:
  - This work extends “Textbooks Are All You Need” from coding to commonsense and general language tasks by training `phi-1.5` largely on synthetic textbook-like content and comparing it to larger open-source models across common-sense, language understanding, and multi-step reasoning (Figures 1; Tables 2–4).

## 3. Technical Approach
- Model architecture (Section 2.1):
  - `phi-1.5` is a Transformer with 24 layers, 32 attention heads, 64 dimensions per head; context length 2048; rotary positional embeddings (“RoPE”) with rotary dimension 32; trained with FlashAttention for speed; tokenizer from `codegen-mono`.
  - Brief definitions:
    - `rotary embedding (RoPE)`: a positional encoding that rotates query/key vectors to encode relative positions, improving extrapolation to longer contexts.
    - `FlashAttention`: an IO-aware attention kernel that computes exact attention faster and with less memory.

- Training data (Section 2.2):
  - Core idea: use mostly synthetic, “textbook-like” data designed to explicitly teach common sense, general knowledge (science, daily activities, theory of mind), and reasoning.
  - Construction:
    - Curated ~20K topic seeds to drive synthetic data generation (~20B tokens).
    - Synthetic generation prompts sometimes included samples from web datasets for diversity, but the final `phi-1.5` training corpus for NLP is synthetic.
    - The only non-synthetic part for `phi-1.5` is ~6B code tokens from `phi-1` (filtered, permissively licensed code), combined with `phi-1`’s prior data to form ~30B raw tokens (Section 2.2).
  - Variants for probing web data:
    - `phi-1.5-web-only`: trained on 95B tokens of filtered web data (88B from Falcon RefinedWeb + 7B code; no synthetic).
    - `phi-1.5-web`: trained on a 40% filtered web + 40% synthetic NLP + 20% code mixture (Section 2.4).

- Training details (Section 2.3; Table 1):
  - Objective: standard next-token prediction (base model; no instruction tuning or RLHF).
  - Optimization: Adam (β1=0.9, β2=0.98, ε=1e−7), constant LR 2e−4, weight decay 0.1, batch size 2048, fp16 with DeepSpeed ZeRO Stage 2.
    - `DeepSpeed ZeRO`: optimizer state/gradient sharding to reduce GPU memory.
  - Tokens seen: 150B training tokens (multiple passes over the ~30B-token corpus), with 80% synthetic and 20% `phi-1` data.
  - Compute and efficiency (Table 1, A100-80G):
    - `phi-1.5`: ~1.5K GPU-hours; inference <3ms/token; ~3.5GB memory at 2048 context.
    - For reference, `Llama-7B` is listed as >80K GPU-hours and ~18GB inference memory.

- Why this approach:
  - Hypothesis from `phi-1`: high-quality, focused “textbook-style” data teaches models to reason and generalize more efficiently than massive, noisy web corpora (Section 2.2 and Discussion).
  - This paper stresses data quality over algorithmic tricks—training is intentionally “straightforward” to isolate data effects (footnote in Section 2.3).

- Evaluation setup (detailed in Section 3 and Figure 1):
  - Benchmarks grouped into:
    - Common sense reasoning: WinoGrande, ARC-Easy/Challenge, BoolQ, SIQA.
    - Language understanding/knowledge: PIQA, HellaSwag, OpenBookQA, SQuAD (EM), MMLU (2-shot).
    - Multi-step reasoning: GSM8K (math word problems), HumanEval/MBPP (code generation).
  - Metrics:
    - Mostly zero-shot accuracy (LM-Eval Harness); SQuAD Exact Match; MMLU 2-shot; code tasks use zero-shot pass@1.
    - `zero-shot`: no task-specific examples in the prompt; `2-shot`: two labeled examples provided.
    - `pass@1`: the single generated solution must be correct.

## 4. Key Insights and Innovations
- Data-quality-over-scale can deliver strong commonsense and reasoning at small scale:
  - `phi-1.5` (1.3B) matches or outperforms 5–10x larger models (e.g., `Vicuna-13B`, `Llama 2-7B`) on many common sense and multi-step reasoning tasks.
  - Evidence:
    - Common sense: “phi-1.5 models perform comparable in common sense reasoning” (Figure 1). Table 2 shows `phi-1.5` on WinoGrande 0.734 and ARC-Challenge 0.444—comparable to `Llama 2-7B` (0.691, 0.434).
    - Multi-step reasoning: “vastly exceeds other models” (Figure 1). Table 4 shows `phi-1.5` gets GSM8K 40.2, HumanEval 34.1, MBPP 37.7—beating many larger baselines.
  - Significance: Suggests careful synthetic curricula can teach reasoning patterns efficiently, reducing reliance on trillion-token web data.

- Synthetic data reduces toxic generations (Section 4; Figure 2):
  - On ToxiGen, `phi-1.5`/`phi-1.5-web` get higher safety scores (closer to 1 is better), across 13 demographics (Figure 2).
  - Manual stress test with 86 adversarial prompts: 
    > “phi-1.5 had a ‘pass’ on 47 prompts, a ‘fail’ on 34, and 4 ‘did not understand’,” while `Llama2-7B` and `Falcon-7B` failed on 54 and 50 prompts, respectively (Section 4).
  - Significance: Training on synthetic textbooks (with minimal raw web data) appears to attenuate memorization of toxic patterns.

- Web–synthetic synergy and clear ablations (Section 2.4; Tables 2–4):
  - `phi-1.5-web-only` (web data only) improves over similar-size web-trained models (e.g., Falcon-rw-1.3B) despite using only ~15% of RefinedWeb (Table 2 discussion).
  - Mixing (`phi-1.5-web`) yields the best multi-step reasoning (e.g., GSM8K 44.6, HumanEval 41.4; Table 4), showing web data still helps in certain reasoning tasks when combined with synthetic curricula.

- Efficiency and accessibility (Table 1):
  - Training compute (~1.5K GPU-hours) and inference memory (~3.5GB at 2k context) make serious reasoning capabilities practical on modest hardware.
  - Significance: Enables broad experimentation (e.g., in-context learning, interpretability) without massive clusters (Discussion, p.14–15).

## 5. Experimental Analysis
- Evaluation methodology and baselines (Section 3; Figure 1; Tables 2–4):
  - Benchmarks:
    - Common sense: WinoGrande (pronoun resolution with commonsense), ARC-Easy/Challenge (multiple-choice science questions), BoolQ (yes/no QA from queries), SIQA (social commonsense).
    - Language/knowledge: PIQA (physical commonsense plausibility), HellaSwag (situational completion with commonsense + knowledge), OpenBookQA (open-book science), SQuAD-EM (reading comprehension exact match), MMLU (broad knowledge exam).
    - Multi-step reasoning: GSM8K (grade-school math), HumanEval/MBPP (Python code synthesis correctness).
  - Metrics and setup:
    - LM-Eval Harness for zero-shot accuracy where applicable; MMLU 2-shot; SQuAD EM; code pass@1. Evaluations come from the paper’s unified pipeline, so absolute numbers may differ slightly from external leaderboards (Figure 1 caption).

- Main quantitative results:
  - Common sense (Table 2):
    > `phi-1.5`: WinoGrande 0.734; ARC-Easy 0.756; ARC-Challenge 0.444; BoolQ 0.758; SIQA 0.526.  
    > `phi-1.5-web`: 0.740; 0.761; 0.449; 0.728; 0.530.  
    - These are on par with `Llama 2-7B` (e.g., ARC-Challenge 0.434, SIQA 0.480) and `Vicuna-13B` (Table 2).
    - Notably, `phi-1.5-web-only` (web-only) already beats other 1–3B models (e.g., Falcon-rw-1.3B) and shows the filtering pipeline is strong even without synthetic data.

  - Language understanding and knowledge (Table 3):
    > `phi-1.5`: PIQA 0.766; HellaSwag 0.476; MMLU 0.376; OpenBookQA 0.372; SQuAD EM 0.72.  
    > `phi-1.5-web`: 0.770; 0.484; 0.379; 0.360; 0.74.  
    - Mixed results: on MMLU, `Llama 2-7B` is higher (0.453), suggesting broader factual knowledge benefits from large-scale web exposure. But `phi-1.5`/`phi-1.5-web` are competitive on PIQA and SQuAD EM (0.72–0.74), even outperforming `Llama 2-7B` on SQuAD EM (0.67).

  - Multi-step reasoning (Table 4):
    > `phi-1.5`: GSM8K 40.2 (“via coding”), HumanEval 34.1, MBPP 37.7.  
    > `phi-1.5-web`: GSM8K 44.6 (“via coding”), HumanEval 41.4, MBPP 43.5.  
    - These outperform many larger baselines. For example, `Llama2-7B` gets 14.6 on GSM8K, 12.8 on HumanEval, 20.8 on MBPP; even `Llama-65B` gets HumanEval 23.7 and MBPP 37.7 (Table 4).
    - Observation (Section 3): mixing web data helps multi-step reasoning more than purely synthetic, likely due to broader algorithmic/code exposure.

  - Safety (Section 4; Figure 2):
    - ToxiGen metric (higher is better): `phi-1.5`/`phi-1.5-web` consistently score above `falcon-rw-1b`, `falcon-rw-7b`, `gpt2-xl`, `opt-1.3b` across 13 demographics (Figure 2).
    - Manual robustness: 
      > Of 86 adversarial prompts, `phi-1.5` had 47 “pass”, 34 “fail”, 4 “did not understand”; `Llama2-7B` and `Falcon-7B` failed on 54 and 50 prompts respectively (Section 4).

- Ablations and trade-offs visible in results:
  - `web-only` vs `synthetic-only` vs `mixed`:
    - `phi-1.5-web-only`: strong for its size (Table 2), but weaker than `phi-1.5` on language understanding and multi-step reasoning (Tables 3–4).
    - `phi-1.5`: strong on common sense and math/coding, with reduced toxicity.
    - `phi-1.5-web`: best on math/coding, indicating web data complements synthetic curricula for algorithmic reasoning (Table 4).

- Qualitative behavior (Section 5):
  - Even as a base model (no instruction tuning), `phi-1.5` follows simple instructions and exhibits chain-of-thought when prompted (“Let’s think step by step”). It sometimes generates overly long continuations or minor code inaccuracies (e.g., mis-explaining `s.bind(('', 0))`), demonstrating both emerging capabilities and base-model rough edges.
  - Examples show commonsense consistency (e.g., adapting a London-in-July scenario to rain), and step-by-step arithmetic reasoning (Section 5).

- Do experiments support the claims?
  - Yes for the central claims:
    - Comparable or better performance than much larger models on common sense and multi-step reasoning (Figures 1, Tables 2–4).
    - Reduced toxicity relative to web-trained baselines (Figure 2; manual evaluations).
    - Efficiency advantages (Table 1).
  - Caveat: Results use the paper’s own evaluation pipeline; while this ensures cross-model consistency within the paper, external numbers may differ slightly (Figure 1 caption).

## 6. Limitations and Trade-offs
- Base model alignment:
  - No instruction tuning or RLHF (Section 2.4 “Remark”; Section 5). As a result, the model:
    - Sometimes doesn’t stop cleanly; may generate long, training-style continuations.
    - Can misinterpret instructions or produce code with minor errors (Section 5 examples).
  - Trade-off: better research utility as a base model vs. less polished user-facing behavior.

- Knowledge coverage vs. reasoning:
  - Mixed results on knowledge-heavy tasks like MMLU (Table 3), where `Llama 2-7B` leads. Synthetic textbooks may not cover as much long-tail factual content as broad web data.

- Toxicity not eliminated:
  - While improved, `phi-1.5` still fails on a non-trivial fraction of adversarial prompts (34/86; Section 4), and base models lack the refusal behavior typical of aligned chat models.

- Data creation effort and bias:
  - High-quality synthetic corpus creation is non-trivial—requires iterative topic selection and gap analysis (Section 2.2). 
  - Synthetic data inherits biases and styles from the generator LLMs and prompt design, which may shape model behavior in hard-to-measure ways.

- Evaluation scope:
  - The paper primarily evaluates English-language benchmarks and entry-level coding/math tasks; real-world robustness (e.g., domain-specific knowledge, multilinguality, tool use) is not targeted.

- “Via coding” protocol on GSM8K:
  - GSM8K performance is reported “via coding” (Table 4), implying a programmatic reasoning approach in prompting. Details of this exact prompting recipe are brief, which could limit replicability without provided prompts.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Demonstrates that carefully curated, synthetic “textbook” data can deliver strong reasoning in compact models, challenging the assumption that only scale unlocks such capabilities (Discussion, p.14–15).
  - Opens a path to capability-per-compute gains: smaller models that are cheaper to train/deploy can still handle substantive reasoning tasks (Table 1).

- Follow-up research enabled or suggested:
  - Data-centric LLM research:
    - Methods for constructing, auditing, and expanding synthetic curricula (coverage analysis, gap-filling, diversity without toxicity).
    - Automated topic selection and iterative dataset refinement loops (Section 2.2 emphasizes this is a core technical skill).
  - Alignment for base models:
    - Instruction tuning/RLHF on top of textbook-trained bases to reduce toxicity and improve refusal behavior, without eroding reasoning ability (Section 4 motivation).
  - Mechanistic interpretability and in-context learning:
    - A 1.3B parameter model with large-model traits is a tractable testbed for mechanistic studies and ICL analyses (Discussion).
  - Safety evaluation frameworks:
    - Extend targeted adversarial tests beyond ToxiGen and the 86-prompt set; measure trade-offs between safety and reasoning after alignment.

- Practical applications:
  - On-device or low-resource deployments requiring strong reasoning (edge devices, private inference contexts).
  - Educational tools that benefit from step-by-step explanations without needing frontier-scale models.
  - Coding assistants for entry-level programming tasks where compactness and safety matter.

> In sum, the paper provides concrete evidence—through architecture transparency (Section 2.1), data construction (Section 2.2), controlled comparisons (Tables 2–4), and safety analyses (Figure 2)—that “textbook-quality” synthetic data can substantially compress the compute and parameter footprint required for high-quality commonsense and reasoning performance. The release of `phi-1.5` as a base model invites the community to test how far this data-centric route can go when combined with alignment, better curricula, and broader evaluation.
