# Tuning Language Models by Proxy

**ArXiv:** [2401.08565](https://arxiv.org/abs/2401.08565)
**Authors:** Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, Noah A. Smith
**Institutions:** University of Washington (implied for Alisa Liu et al.)

## 🎯 Pitch

Proxy-tuning introduces an innovative method to customize large language models by adjusting their token scores through a small, tuned model's influence rather than altering the base model's weights. This approach significantly reduces computational overhead and democratizes model customization, allowing users to adapt proprietary or costly models with only black-box access, preserving pretraining knowledge and offering flexible control over model behavior.

---

## 1. Executive Summary
Proxy-tuning is a decoding-time method that “tunes” a large language model (LM) without touching its weights by adding an offset to its next-token scores derived from a small tuned model and its untuned counterpart. Across instruction following, safety, reasoning, coding, and task-specific finetuning, this simple offset closes most of the gap to fully finetuned chat models while requiring only black‑box access to token probabilities (see §2 and Eq. (1)).

## 2. Context and Motivation
- Problem addressed
  - Organizations and users often need to adapt large LMs to new behaviors (instruction following, safety, domains like code, or specific tasks). Direct finetuning is expensive and sometimes impossible for proprietary models that hide their weights. The core gap is: how to effectively customize a strong LM using only its output probabilities/logits at inference time.
- Why it matters
  - Practical impact: Enables adapting closed or costly models (e.g., 70B+ or commercial APIs) without retraining. This reduces compute cost, democratizes customization, and helps preserve the model’s pretraining knowledge that can be harmed by finetuning (“alignment tax”).
  - Scientific interest: Tests whether “tuning signals” learned in small models can transfer to large models via decoding-time composition, informing our understanding of what alignment/finetuning changes in an LM.
- Prior approaches and their shortcomings
  - Parameter-efficient finetuning (e.g., LoRA/QLoRA) still requires white-box access to model weights and significant resources (see Appendix D, Tables 13–15, which show hours-to-days of training even with LoRA).
  - Prompt engineering and long “alignment prompts” can elicit behaviors but are unwieldy at inference (long contexts, shorter remaining budget) and less reliable (§8).
  - Prior controllable-generation via logits (e.g., DExperts, GeDi) targets narrow attributes (toxicity, sentiment) and/or requires training extra components; they do not show a general, black-box route to emulate full instruction tuning across diverse tasks (§8).
- Positioning
  - This work reframes DExperts-style logit arithmetic as a general recipe to emulate finetuning: steer a large base model using the “difference” learned by a small tuned model relative to its untuned version. It demonstrates generality across instruction following (§3), domain adaptation to code (§4), task finetuning (§5), a black-box API case (§7), and analyses that reveal what changes in the token distribution (§6).

## 3. Technical Approach
The mechanism runs entirely at decoding time (no weight updates).

- Key objects and terms
  - `logits`: the raw, unnormalized scores an LM assigns to each token before applying softmax to get probabilities.
  - `expert` (`M+`): a small model that has been tuned for a desired behavior (e.g., instruction-following).
  - `anti-expert` (`M-`): the same small model family but untuned (pretrained base).
  - `base model` (`M`): the large, powerful LM we want to steer without access to its weights.

- Step-by-step procedure (see §2 and Figure 1)
  1. At each generation step t, run the prompt prefix `x_<t` through three models:
     - Large base `M` → logits `s_M(· | x_<t)`.
     - Small tuned expert `M+` → logits `s_M+(· | x_<t)`.
     - Small untuned anti-expert `M-` → logits `s_M-(· | x_<t)`.
  2. Compute a logit offset equal to the expert–anti-expert difference: `Δ = s_M+ − s_M-`. Intuition: this isolates “what tuning changed” in the small model.
  3. Add this offset to the base model’s logits and sample the next token from the resulting distribution:
     - Equation (1): 
       > p_M̃(X_t | x_<t) = softmax[s_M + s_M+ − s_M-]
  4. Repeat for subsequent tokens.

- Why this works
  - The difference `s_M+ − s_M-` encodes the tuning effect learned at small scale. Adding it to `s_M` nudges the large model toward the tuned behavior while retaining its pretraining knowledge (Figure 1 shows real logit shifts on a TruthfulQA example).
  - Alternate view (under Eq. (1)): `s_M+ + (s_M − s_M-)` looks like giving the small expert the benefit of large-model pretraining via contrastive decoding while still injecting the expert’s tuning.

- Design choices and practicality
  - Vocabulary alignment: Models must share a tokenizer vocabulary. This is easy for open LMs in the same family (e.g., LLaMA2 and CodeLlama share tokenization; §4.1). If not, one can map tokenizations (footnote in §2).
  - No extra hyperparameters by default, but an optional strength parameter `α` allows control: `s_M + α(s_M+ − s_M-)`. Figure 2 shows a smooth trade-off between truthfulness and informativeness on TruthfulQA (§6.2).
  - Decoding setup: For instruction tasks the work uses zero-shot prompts and greedy decoding (§3.1); for code, it uses standard sampling settings (top‑p=0.95, temperature=0.8) to evaluate pass@10 (§4.2).
  - Extremely limited black-box setting (GPT‑3.5): only top‑5 token log-probabilities are available, and the API cannot be conditioned on partial responses (§7). The work confines steering to multiple‑choice questions by only reweighting the four option tokens {A,B,C,D} using the offset from small (anti-)experts.

- Computational aspects
  - Runtime overhead comes from running three forwards (base, expert, anti-expert) per step. Appendix C.1 (Table 12) measures ~1.5×–2.5× slowdown when run sequentially, with a clear path to parallelization across GPUs to approach tuned-model latency.

## 4. Key Insights and Innovations
- A general decoding-time emulator for finetuning
  - Novelty: Using the expert–anti-expert logit difference from a small model to steer a large, untuned base model (Eq. (1), §2; Figure 1). Unlike prior controllable-generation methods, this targets the broad outcome of “being tuned” rather than a single attribute.
  - Significance: It closes most of the gap to fully finetuned chat models on instruction, reasoning, and safety while requiring only black-box access to probabilities (Table 2).

- Knowledge preservation and style/behavior control
  - Insight from §3 and Table 3: On TruthfulQA (open-ended), proxy-tuned models are slightly less informative but more truthful than fully finetuned chat models:
    > 70B: informative 92.8% vs 93.8% (Chat), truthful 92.3% vs 85.8% (Chat) (Table 3).
  - Interpreted with §6.1 token analyses: proxy-tuning most strongly boosts tokens that express reasoning style and caution (“There is no scientific…”, “is a common myth”, “I cannot provide”), suggesting instruction-tuning largely affects style and reasoning scaffolding rather than raw knowledge.

- Versatility across settings, including API-only models
  - The framework works for instruction following (§3), code domain adaptation (§4), strict-format task finetuning (§5), and an API with only top‑5 log‑probs (GPT‑3.5; §7). Example: 
    > GPT‑3.5 accuracy on REALTIMEQA improves from 54.2% to 56.5% with proxy‑tuning (Table 7).

- On-demand controllability without retraining
  - The optional strength knob `α` in §6.2 lets users trade “helpfulness” for “truthfulness” at runtime:
    > Increasing `α` monotonically improves truthfulness on TruthfulQA while eventually hurting informativeness (Figure 2).
  - This per-request control is hard to achieve with weight finetuning.

## 5. Experimental Analysis
- Evaluation setup (details in §3–§7 and Appendix A)
  - Instruction-following and safety:
    - Datasets: `AlpacaFarm` (win rate vs text-davinci-003 judged by GPT‑4), `GSM` (math word problems, accuracy), `ToxiGen` (should not continue toxicity; metric: % toxic), `TruthfulQA` (open-ended: % informative and truthful; multiple-choice also used).
    - Models: Base LLaMA2‑13B or 70B as `M`; `M+` = LLaMA2‑7B‑CHAT; `M-` = LLaMA2‑7B‑BASE (§3).
    - Decoding: zero-shot, greedy; TruthfulQA Chat models receive a system prompt (Appendix A.1, Table 9).
  - Code domain adaptation (§4):
    - Datasets: `CodexEval` (HumanEval) and `DS-1000` (data science problems). Metric: pass@10 with sampling (Appendix A.2).
    - Expert: `CODELLAMA‑7B‑PYTHON` as `M+`; `M-` = LLaMA2‑7B‑BASE.
  - Task-specific finetuning (§5):
    - Tasks: `TriviaQA` (exact match), `GSM` (strict answer format).
    - Experts: LLaMA2‑7B tuned on each task; also trained 13B and 70B task experts for comparison.
  - Black-box API (§7):
    - Dataset: `REALTIMEQA` (news-like questions with time-sensitive answers).
    - Base: `gpt-3.5-turbo-0613` (only top‑5 log‑probs per step).
    - Steering: reweights only {A,B,C,D} tokens using LLaMA2‑7B‑based (anti-)experts retrained on retrieved articles.

- Main quantitative results
  - Instruction-following (Table 2; highlights):
    > LLaMA2‑70B BASE vs Proxy vs Chat on AlpacaFarm: 3.7% → 88.0% → 90.4% win rate.  
    > GSM (accuracy): 9.6% → 32.0% → 51.8%.  
    > ToxiGen (% toxic): 67.4% → 0.0% → 0.0%.  
    > TruthfulQA (% Info+True): 53.9% → 85.1% → 79.6%.
    - “Gap closed” metric averaged over tasks: 
      > 91.1% at 13B and 88.1% at 70B (§3.2).
    - Note the striking safety result: toxicity drops from ~67–70% to ~0% at both 13B and 70B.
    - On knowledge‑stress test (TruthfulQA open-ended), proxy-tuning even surpasses Chat in truthfulness (Table 3).
  - Code domain adaptation (Table 4):
    > 13B CodexEval pass@10: 33.7% → 65.7% (proxy) vs 68.9% (7B code-tuned expert).  
    > 13B DS‑1000 pass@10: 26.2% → 42.8% vs 53.6%.  
    > 70B CodexEval: 62.0% → 70.7% vs 89.2% (CodeLlama‑70B‑Python, tuned).  
    > 70B DS‑1000: 43.9% → 50.6% vs 67.6%.
    - Interpretation in §4.2: the contrast `(13B‑BASE – 7B‑BASE)` does not help the 7B code expert; domain-specialized pretraining appears to dominate.
  - Task finetuning (Table 5):
    > 13B TriviaQA EM: 36.8% → 55.9% (proxy) vs 59.5% (13B task‑tuned).  
    > 13B GSM: 6.6% → 43.9% vs 51.0%.  
    > 70B TriviaQA: 45.2% → 62.7% vs 63.1%.  
    > 70B GSM: 9.6% → 53.9% vs 67.9%.
    - Proxy-tuning “inherits” strict formatting from the small expert: for GSM, >99.7% of proxy outputs place the final answer after “####” as required (§5.2).
  - Black-box GPT‑3.5 temporal adaptation (Table 7):
    > REALTIMEQA accuracy: 54.2% (base) → 56.5% (proxy-tuned), with p < 0.0001 (§7).
  - Analyses (Section 6 and Appendix C):
    - Token-level influence (§6.1): On GSM, the probability boost `Δ_t` is more than 2× larger on the left-hand-side of intermediate equations than on right-hand-side numerals
      > LHS mean 0.131 vs RHS 0.056; p < 0.0001.
    - Most boosted tokens on TruthfulQA (Table 6) include contexts like “There is no scientific…”, “is a common myth”, “I cannot provide”, consistent with style/caution gains.
    - Strength knob (Figure 2, §6.2): increasing `α` improves truthfulness monotonically but can reduce informativeness.
    - Runtime (Appendix C.1, Table 12): sequential proxy-tuning is ~1.5× (70B) to ~2.5× (13B) slower per generation than a single tuned model; parallel execution across GPUs can mitigate this.
    - How often predictions change (Appendix C.2): proxy-tuning changes the base model’s top token ~13–25% of the time, with the largest effect on early tokens (Figure 3).

- Do the experiments support the claims?
  - Strong evidence for instruction-following and safety: large, consistent gains across four datasets and two scales (Table 2) with near-zero toxicity and improved truthfulness.
  - Evidence for generality: positive results in code and task finetuning, though code results reveal limits (proxy often trails a specialized code-tuned model; Table 4).
  - Black-box feasibility: small but statistically significant improvement for GPT‑3.5 in an extremely constrained top‑5‑logit setting (Table 7), demonstrating practical viability.

- Additional comparisons with LoRA (Appendix D)
  - Performance (Table 15): LoRA sometimes excels (TriviaQA) and sometimes underperforms (GSM at 13B). Proxy-tuning beats LoRA on GSM at 13B by +11.5 points but trails at 70B by −9.1.
  - Training efficiency (Table 14): Training a 7B expert fully (for proxy-tuning) took ~30h for TriviaQA vs LoRA‑13B at ~34h and LoRA‑70B at ~459h. For GSM the differences are starker (2h35m vs 3h49m vs 39h20m). This underscores a cost advantage when you can reuse a small expert.

## 6. Limitations and Trade-offs
- Access assumptions
  - Requires access to next-token logits or probabilities from the base model. Many APIs expose top‑k only; §7 shows a workaround for multiple-choice but not for open‑ended generation.
  - Requires shared or mappable vocabularies across models (§2, footnote). Tokenizer mismatches add complexity.
- Computational overhead
  - Increases inference cost by running three models each step (Appendix C.1, Table 12). Parallelization helps but needs extra hardware coordination.
- When it underperforms
  - Domain‑specialized settings (code): the proxy may not surpass a directly specialized model at larger scales (Table 4). The contrast `(large BASE – small BASE)` can contribute little when domain specialization dominates (§4.2).
- Behavioral trade-offs
  - On TruthfulQA, proxy-tuning increases truthfulness but can slightly reduce informativeness (Table 3; Figure 2). The `α` knob mitigates this but requires tuning per application (§6.2).
- Scope of control and compositionality
  - The method relies on the quality and availability of an appropriate small “expert.” If suitable experts are lacking or misaligned with the target behavior, steering may be suboptimal.
- Black-box constraints
  - In settings like GPT‑3.5 where only top‑k log‑probs are returned and partial-response conditioning is unavailable, the approach cannot be applied to open-ended generation (§7).

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes decoding-time tuning as a practical alternative to weight finetuning for large or proprietary LMs. It lowers barriers to customization, encourages providers to expose token probabilities, and suggests that “alignment” can often be injected as a stylistic/logit-level correction rather than weight updates.
- Enabled research directions
  - Composition of multiple experts (safety + domain + task) with principled weighting schemes; automatic selection of `α` per prompt.
  - Improved logit‑mapping across tokenizers to broaden cross-family steering (noted in §2 footnote).
  - Adaptive or selective application across timesteps (Appendix C.2 shows early tokens matter most), reducing runtime while retaining quality.
  - Understanding “alignment tax”: further studies on knowledge retention vs stylistic alignment, extending the token‑level analyses of §6.1.
  - Weak‑to‑strong transfer: more systematic study of how small weak experts can steer stronger bases (as hinted by the GPT‑3.5 case; §7).
- Practical applications
  - Rapidly align base models for safety and refusal behaviors (ToxiGen drops to 0% toxicity; Table 2).
  - Temporally update knowledge for proprietary models when weight updates are unavailable (REALTIMEQA improvements, §7).
  - Enforce strict output formats in evaluation or production pipelines (GSM “#### final answer” adherence; §5.2).
  - Domain adaptation when direct access is limited—e.g., enterprise settings where model weights are restricted but logits can be exposed.

Overall, proxy-tuning is a simple, general, and surprisingly powerful technique: by adding the small expert’s tuning “delta” to a large base model’s logits (Eq. (1), Figure 1), it recovers most of the benefits of full finetuning, preserves knowledge, and works even with highly constrained black-box APIs.
