# Camels in a Changing Climate: Enhancing LM Adaptation with TÜLU 2

**ArXiv:** [2311.10702](https://arxiv.org/abs/2311.10702)

## 🎯 Pitch

TÜLU 2 delivers an open, reproducible suite of large language models—spanning up to 70B parameters—finetuned on a rigorously curated instruction dataset and enhanced with scalable Direct Preference Optimization (DPO). This work not only sets a new open benchmark for instruction-following and coding ability, but also demystifies what matters in adapting LLMs to downstream tasks, providing the open-source community with all the models, data, and recipes needed for rapid and reliable language model alignment.

---

## 1. Executive Summary

This paper builds on the original TÜLU instruction-tuning framework by systematically evaluating and incorporating recent advances in base models, instruction datasets, and adaptation methods into **TÜLU 2**, a suite of open LLAMA-2 and CODE LLAMA models fine-tuned across 7B, 13B, and 70B parameter scales. The core mechanism is a redesigned data mixture (**TÜLU-V2-mix**) that combines high-quality sources—including new distilled datasets like WizardLM Evol-Instruct V2 (complexity-diverse GPT-4 generations) and Open-Orca (augmented FLAN explanations)—with a streamlined set of retained human datasets, yielding an average 8% improvement over the original V1 mixture at the 7B scale. A second mechanism is **direct preference optimization (DPO)** applied post-fine-tuning using UltraFeedback data, which boosts open-ended generation quality—improving AlpacaEval win rates by an average of 13% across model sizes—and scales stably to 70B parameters, producing the largest publicly released DPO-trained model to date (TÜLU 2+DPO 70B, achieving 95.1% AlpacaEval and 7.89 MT-Bench, state-of-the-art among open-weight models). The paper also establishes that QLoRA parameter-efficient fine-tuning underperforms full fine-tuning on long-form generation tasks—by 20% on AlpacaEval—with the gap shrinking as model size increases, and that adapting CODE LLAMA base models with the V2 mixture produces CODE TÜLU 2 models that outperform both base CODE LLAMA and CODE LLAMA-Instruct on coding benchmarks, though at a roughly 20% cost to AlpacaEval performance, establishing that continued code pretraining significantly reshapes model capabilities in non-code domains.

## 2. Context and Motivation

### The Core Problem: Open Instruction-Tuning Recipes Are Lagging Behind Proprietary Systems

The fundamental problem this paper addresses is deceptively simple: **how do we build open-source language models that approach the instruction-following capabilities of proprietary systems like GPT-4 and GPT-3.5-turbo?** Six months before this paper's release, the original TÜLU work (Wang et al., 2023b) established that openly available base models, datasets, and fine-tuning methods could produce competent instruction-tuned models, but the landscape was evolving at extraordinary speed. The central question is not whether open models *can* be instruction-tuned—TÜLU 1 already demonstrated that—but rather: **what specific combinations of base models, data mixtures, and adaptation methods work best, and how do we systematically improve these recipes as new components become available?**

This matters for both practical and scientific reasons:

- **Accessibility and reproducibility**: Proprietary systems like GPT-4 are trained on undisclosed data with undisclosed methods. Every advance in open instruction tuning makes the technology more accessible to researchers, smaller organizations, and applications where sending data to external APIs is infeasible (privacy-sensitive domains, offline deployment, cost-sensitive applications).
- **Scientific understanding of adaptation**: The relationship between training data composition, model scale, and downstream behavior is not well characterized. Does instruction data quality matter less as models get larger? Does RLHF-style training (via DPO) preserve or degrade multilingual capabilities? How does continued pretraining on specialized data (code) affect general instruction-following? These mechanistic questions require systematic study with fully open artifacts.
- **Rapid iteration**: The field was producing new base models (LLAMA-2, CODE LLAMA, MISTRAL), new training methods (DPO, QLoRA), and new datasets (UltraFeedback, Open-Orca, WizardLM) at a pace that made it impossible for practitioners to know which combinations were worth adopting. A comprehensive evaluation combining these advances was needed.

### The Gap: No Systematic Integration of Recent Advances

The original TÜLU work established a baseline—LLAMA models fine-tuned on a mixture of human-written and model-generated instruction data—and showed that open models could approach early GPT-3.5 variants on several benchmarks. But since that release, the components available for building instruction-tuned models had improved substantially along three independent axes:

**Better base models**: LLAMA-1 had been superseded by LLAMA-2 (Touvron et al., 2023b), which was pretrained on roughly twice as many tokens (2 trillion vs. 1–1.4 trillion) and showed an average 10% improvement across academic benchmarks. CODE LLAMA (Roziere et al., 2023) extended LLAMA-2 with additional code pretraining, creating models specialized for programming tasks. MISTRAL (Jiang et al., 2023) introduced architectural innovations at the 7B scale. Each of these base models had different strengths, and how they would interact with different instruction-tuning recipes was unknown.

**Better instruction datasets**: The data landscape had shifted dramatically. The original TÜLU V1 mix relied heavily on FLAN (Chung et al., 2022), Dolly (Databricks, 2023), and a smaller set of GPT-generated data. Since then, several high-quality datasets had emerged:
- **WizardLM Evol-Instruct V2** (Xu et al., 2023) introduced a method for automatically generating increasingly complex and diverse instructions through iterative GPT-4 prompting, producing data that pushed models toward more sophisticated instruction-following.
- **Open-Orca** (Lian et al., 2023) replicated the Orca approach (Mukherjee et al., 2023) of augmenting FLAN tasks with detailed GPT-4 explanations, providing richer training signals.
- **LIMA** (Zhou et al., 2023) demonstrated that remarkably strong alignment could be achieved with only ~1,000 carefully hand-curated examples, challenging the assumption that large datasets were necessary.
- **UltraChat** (Ding et al., 2023) and **UltraFeedback** (Cui et al., 2023) provided large-scale conversation and preference data specifically designed for fine-tuning and RLHF.

These datasets had been studied individually in their respective papers, but no one had systematically evaluated which combinations work well together, whether benefits compound, or how their effects scale with model size.

**Better fine-tuning methods**: Perhaps the most consequential methodological advance was **Direct Preference Optimization (DPO)** (Rafailov et al., 2023). Before DPO, adding RLHF-style training to instruction-tuned models required the complex PPO pipeline: train a separate reward model, then use it to score generations during online RL training, requiring multiple models in memory and careful hyperparameter tuning to prevent reward hacking. DPO simplified this dramatically by reparameterizing the preference optimization as a direct supervised loss on preference pairs—no separate reward model, no online sampling, no PPO. Zephyr-Beta (Tunstall et al., 2023) had shown that DPO applied to a fine-tuned MISTRAL-7B model using UltraFeedback data produced dramatic improvements in open-ended generation. But Zephyr-Beta only tested this recipe at the 7B scale with one specific base model. It was unknown whether DPO training would remain stable at larger scales (13B, 70B), whether the Zephyr-Beta hyperparameters would transfer, or whether DPO would complement or conflict with different supervised fine-tuning data mixtures.

Simultaneously, **QLoRA** (Dettmers et al., 2023) promised to dramatically reduce the computational cost of fine-tuning by combining 4-bit quantization with low-rank adapters. If QLoRA could match full fine-tuning performance, it would democratize instruction tuning to researchers without access to large GPU clusters. But Dettmers et al. (2023) primarily validated QLoRA on MMLU, leaving open the question of whether the method worked for the broader evaluation suite (including open-ended generation) that instruction-tuned models are judged on.

### Where Prior Approaches Fall Short

The paper identifies specific limitations in how these advances had been studied:

**Fragmented evaluation**: Each new base model, dataset, or training method was typically evaluated in isolation, on different benchmarks, against different baselines. Zephyr-Beta reported AlpacaEval and MT-Bench scores. The original LLAMA-2 paper reported primarily on safety and MMLU. WizardLM reported on a custom test set of complex instructions. This made it impossible to draw head-to-head conclusions. Is Zephyr-Beta better than LLAMA-2-Chat? Is the WizardLM dataset better than Open-Orca? Without a unified evaluation framework, practitioners were left guessing.

**No scaling analysis for DPO**: The Zephyr-Beta recipe—supervised fine-tuning on UltraChat followed by DPO on UltraFeedback—had only been demonstrated at 7B. The literature contained no examples of DPO training at 13B, 34B, or 70B scales. This was not merely an engineering question; there were theoretical reasons to worry. DPO optimizes a loss function that is mathematically equivalent to RLHF under certain assumptions, but at larger scales, optimization dynamics could change—the model might overfit to preference data, exhibit training instability, or show qualitatively different downstream effects. The paper explicitly notes the absence of evidence: "To our knowledge, TÜLU 2+DPO 70B is the largest publicly-released DPO-trained model" (Section 3.3).

**Unclear QLoRA boundaries**: QLoRA had been validated on MMLU, but instruction-tuned models are evaluated on a much broader set of capabilities—multilingual QA, coding, open-ended generation, toxicity. Dettmers et al. (2023) had not tested whether QLoRA's approximation error disproportionately affected certain task types. The paper hypothesizes that open-ended generation might be particularly sensitive because it requires the model to produce coherent, diverse, and stylistically appropriate long-form text—a capability that might depend on the full precision of the fine-tuning updates.

**Domain specialization vs. general capability**: CODE LLAMA represented an important step toward domain-specialized base models, but the tradeoffs were unknown. Continued pretraining on code was designed to improve programming, but what did it cost in other domains? Did code-adapted models lose general instruction-following ability, factual knowledge, or multilingual capability? The CODE LLAMA paper focused primarily on coding benchmarks, leaving cross-domain effects unexplored.

**Dataset interaction effects**: The original TÜLU work showed that combining multiple datasets often helped, but that no single dataset was universally beneficial—some tasks improved while others degraded. With the new datasets (WizardLM, Open-Orca, LIMA), the interaction patterns were unknown. Would adding a heavily distilled dataset like WizardLM complement or conflict with human-written data from FLAN? Would LIMA's small but carefully curated examples provide disproportionate benefits when mixed with larger noisier datasets?

### How This Paper Positions Itself

The paper frames itself not as proposing fundamentally new methods, but as **systematically integrating and evaluating recent advances that had emerged since TÜLU 1**. This is evident from the structure: the "advances" in Section 2 are all adopted from other work—LLAMA-2 base models (Touvron et al., 2023b), DPO training (Rafailov et al., 2023; Tunstall et al., 2023), QLoRA (Dettmers et al., 2023), CODE LLAMA (Roziere et al., 2023). The novel contribution is the **integration recipe**—which components to combine, at what scales, with what data mixture—and the **comprehensive evaluation** that characterizes when and why each component helps.

This positioning is valuable because the field suffers from a proliferation of incomparable results. The paper establishes a fixed evaluation framework (MMLU, GSM8k, BBH, TydiQA, CodexEval, AlpacaEval, ToxiGen, TruthfulQA, MT-Bench) and tests all variants within it, producing claims that are directly comparable: "V2 mix outperforms V1 mix by 8% on average at 7B" or "DPO improves AlpacaEval by 10-12% across scales." These numbers, while specific to the LLAMA-2 family and the V2 mixture, provide anchor points that subsequent work can calibrate against.

The paper also explicitly connects to the broader trend of **open-weight models competing with proprietary systems**. The abstract notes that TÜLU 2 "matches or exceeds the performance of GPT-3.5-turbo-0301 on several benchmarks," and Table 1 shows TÜLU 2 70B achieving comparable MMLU (67.3 vs. 67.9), BBH (68.4 vs. 66.1), and superior AlpacaEval (86.6 vs. 83.6). This framing—open models catching up to proprietary ones—is not new (Vicuna, Zephyr, Xwin-LM, and LLAMA-2-Chat all made similar claims), but TÜLU 2 provides the most systematic ablation of the recipe components that contribute to that catch-up.

A subtle but important positioning choice: the paper does **not** claim to train a chatbot. The models are instruction-tuned for task completion, not optimized for conversational safety or refusal behavior. This distinguishes TÜLU 2 from LLAMA-2-Chat (which underwent extensive safety RLHF) and positions it more as a research artifact for studying adaptation methods than as a deployment-ready assistant. The ToxiGen results (Table 1) confirm this—TÜLU 2 70B scores 0.5% toxic while LLAMA-2-Chat 70B scores 0.0%—but the paper does not frame toxicity reduction as a primary goal. This scoping is important because it means TÜLU 2's strong AlpacaEval and MT-Bench numbers reflect pure instruction-following capability rather than carefully engineered safety tradeoffs.

## 3. Technical Approach

### 3.1 Reader Orientation

The TÜLU 2 project is a **systematic recipe integration and evaluation pipeline** — not a single novel algorithm — that takes a pretrained base language model, a curated mixture of instruction-following datasets, and a sequence of fine-tuning stages to produce instruction-tuned models spanning 7B to 70B parameters. The problem it solves is: given the rapid proliferation of new base models, instruction datasets, and adaptation methods, how do we combine them into a coherent recipe that produces the strongest possible open-weight instruction-following models? The shape of the solution is a **modular experimental framework** where each component (base model, data mixture, fine-tuning method, optional DPO stage) can be swapped independently, evaluated on a fixed battery of benchmarks, and compared head-to-head, yielding concrete empirical claims about which combinations work best and why.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components arranged in a pipeline:

1. **Base Model Selection** — Choose from LLAMA-2 (general-purpose) or CODE LLAMA (code-specialized) at 7B, 13B, 34B, or 70B scales. This determines the raw capabilities that instruction tuning will shape.

2. **TÜLU-V2-mix Data Construction** — Assemble a carefully curated 326,154-example instruction dataset by combining, filtering, and downsampling from eleven sources (FLAN, CoT, Open Assistant 1, ShareGPT, GPT4-Alpaca, Code-Alpaca, LIMA, WizardLM Evol-Instruct V2, Open-Orca, Science Literature, and Hardcoded prompts). This dataset defines what behaviors the model should learn.

3. **Supervised Fine-Tuning (SFT)** — Full-parameter fine-tune the base model on TÜLU-V2-mix for 2 epochs with a maximum sequence length of 8,192 tokens, using a learning rate of `$2 \times 10^{-5}$` (or `$1 \times 10^{-5}$` for 70B). This produces the core TÜLU 2 models.

4. **Direct Preference Optimization (DPO)** — Optionally apply a second training stage using the UltraFeedback preference dataset for 3 epochs with a very low learning rate of `$5 \times 10^{-7}$` and `$\beta = 0.1$`. This biases the model toward outputs that human raters (or GPT-4, in the case of UltraFeedback) prefer, improving open-ended generation quality. This produces the TÜLU 2+DPO variants.

5. **Evaluation Suite** — Evaluate all model variants on a fixed battery: MMLU (factual knowledge), GSM8k (math reasoning), BBH (complex reasoning), TydiQA (multilingual QA), CodexEval (code generation), AlpacaEval and MT-Bench (open-ended generation judged by GPT-4), ToxiGen (toxicity), and TruthfulQA (truthfulness). Results are reported per-model-size, per-training-stage, and compared against proprietary baselines (GPT-3.5-turbo-0301, GPT-3.5-turbo-0613, GPT-4-0613) and competing open models (Zephyr-Beta, Xwin-LM, LLAMA-2-Chat).

Information flows linearly through the pipeline: base model → SFT on V2-mix → (optional) DPO on UltraFeedback → evaluation. Each stage is independent enough that alternative choices (QLoRA instead of full fine-tuning, CODE LLAMA instead of LLAMA-2) can be tested by swapping exactly one component and rerunning evaluation.

### 3.3 Roadmap for the Deep Dive

- **First**, the **TÜLU-V2-mix data construction** — which datasets are included, why specific datasets were added or removed relative to V1, how downsampling and filtering decisions were made, and what the resulting length and composition statistics look like. This is the foundation: everything else depends on what the model is trained on.

- **Second**, the **supervised fine-tuning (SFT) stage** — hyperparameters, training infrastructure, sequence length decisions, and the rationale for full fine-tuning as the default (with QLoRA studied separately as a cost-reduction alternative). This is where the base model is transformed into an instruction-following model.

- **Third**, the **Direct Preference Optimization (DPO) stage** — the mathematical formulation, the UltraFeedback dataset, the specific hyperparameters that enabled stable training at 70B, and the mechanism by which DPO improves open-ended generation without a separate reward model. This is the most technically novel component, and understanding why the learning rate must be `$5 \times 10^{-7}$` (three orders of magnitude lower than SFT) is key.

- **Fourth**, the **QLoRA parameter-efficient alternative** — how it works, what hyperparameters were used, and why it was evaluated as a drop-in replacement for full fine-tuning rather than as a complementary method. This clarifies the cost-performance tradeoff.

- **Fifth**, the **evaluation framework** — what each benchmark measures, how scores are computed, what GPT-4 versions were used as judges, and what contamination issues exist (TruthfulQA prompts in UltraFeedback). This is the lens through which all model variants are compared.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **empirical integration and evaluation paper** whose core idea is that systematically combining recent advances in base models, instruction datasets, and adaptation methods — and evaluating every combination on a fixed benchmark suite — yields a recipe for open-weight instruction-tuned models that approach proprietary performance, while also revealing scaling patterns and tradeoffs that single-component papers miss.

---

#### TÜLU-V2-mix Data Construction

The data mixture is the most consequential design choice in the pipeline because it defines the distribution of behaviors the model learns during supervised fine-tuning. The V2 mixture was constructed through an iterative process of retaining high-performing datasets from V1, adding new datasets that had shown strong results in the literature, and removing or downsampling datasets that underperformed in prior ablations.

**Datasets retained from V1:**

- **FLAN v2** (Chung et al., 2022): A large collection of task-oriented instruction examples spanning diverse NLP tasks. The V2 mixture uses 50,000 examples sampled from FLAN v2, significantly downsampled from the original V1 usage. This downsampling reflects a deliberate shift in dataset philosophy: V1 used FLAN as a primary component, but V2 treats it as one of many equally-weighted sources, reducing its influence in favor of distilled datasets that had shown stronger open-ended generation performance.

- **CoT subset of FLAN v2**: An additional 50,000 examples specifically from the chain-of-thought subset of FLAN v2. The paper explicitly states this is "to emphasize chain-of-thought (CoT) reasoning" — a targeted intervention to preserve reasoning capabilities that might otherwise be diluted by the increased proportion of conversational and open-ended data in V2. This is important because chain-of-thought reasoning requires the model to produce intermediate reasoning steps before the final answer, and training on examples that demonstrate this pattern is necessary to elicit it at test time.

- **Open Assistant 1** (Köpf et al., 2023): A dataset of human-generated conversations from a crowdsourced effort. The V2 mixture uses 7,708 examples, but applies a quality filter: only the "highest-scoring paths in each conversation tree" are retained, where scores come from quality labels provided by the original human annotators. This filtering is crucial because Open Assistant conversations are trees — each prompt can have multiple responses, each response can spawn multiple follow-ups, creating many paths through the tree. By selecting only the highest-quality paths, the dataset avoids training the model on low-quality or contradictory human responses.

- **ShareGPT**: 114,046 examples of conversations between users and ChatGPT (GPT-3.5 or GPT-4), collected from users who voluntarily shared their chat logs. The paper uses a reproduced version of the dataset since the "exact dataset has not been released." The processing follows Vicuna's approach: long conversations are split into blocks with a maximum length of 4,196 tokens. ShareGPT is retained because it had shown strong performance in prior work — particularly for open-ended conversational ability — despite being entirely model-generated.

- **GPT4-Alpaca** (Peng et al., 2023): 20,000 examples generated by GPT-4 using the Alpaca prompting format (instruction-input-output triples). This is downsampled from the full dataset, similar to FLAN.

- **Code-Alpaca** (Chaudhary, 2023): All 20,022 examples from the Code Alpaca dataset, which applies the Alpaca format to code generation tasks. This is retained in full to maintain coding ability.

**Datasets newly added in V2:**

- **LIMA** (Zhou et al., 2023): 1,030 carefully hand-curated examples. LIMA's core finding was that a small number of extremely high-quality examples could produce strong alignment, challenging the assumption that instruction tuning requires massive datasets. Including LIMA in the V2 mixture adds a small but high-signal component that may improve the quality of model outputs even though it represents only ~0.3% of the total training data. The paper does not ablate LIMA's individual contribution, but its inclusion reflects the hypothesis that quality and diversity are more important than quantity.

- **WizardLM Evol-Instruct V2** (Xu et al., 2023): 30,000 examples generated through an iterative process where GPT-4 is prompted to rewrite instructions to make them increasingly complex and diverse. The "Evol-Instruct" method starts with simple seed instructions and repeatedly applies operations like "add constraints," "deepen the reasoning," "increase the number of steps," or "add a confusing element." The resulting dataset contains instructions of substantially higher complexity than typical human-written or single-pass GPT-generated data. The paper samples 30,000 examples from this dataset, making it one of the larger components of V2, reflecting its importance for pushing the model toward sophisticated instruction-following.

- **Open-Orca** (Lian et al., 2023): 30,000 examples generated by GPT-4, replicating the Orca approach (Mukherjee et al., 2023) of taking FLAN tasks and augmenting them with detailed explanations. Orca's insight was that training on not just the answer but the *reasoning process* that GPT-4 uses to arrive at the answer produces stronger student models than training on answers alone. Open-Orca is an open reproduction of this approach, and its inclusion provides another source of rich reasoning traces.

- **Science Literature**: 7,544 examples from a mixture of scientific document understanding tasks, broken down in Appendix C (Table 7) as:
  - Evidence Inference (Lehman et al., 2019): 1,678 examples of extracting medical evidence 5-tuples (intervention, comparator, outcome, sample size, result) from clinical trial reports.
  - Qasper (Dasigi et al., 2021): 2,255 examples of question answering over NLP papers.
  - SciERC (Luan et al., 2018): 700 examples of named entity recognition and relation extraction from scientific text.
  - SciFact (Wadden et al., 2020): 919 examples of fact-checking scientific claims against research literature.
  - SciTLDR (Cachola et al., 2020): 1,992 examples of extreme summarization of scientific documents (producing one-sentence summaries).
  
  This component adds domain-specific scientific reasoning capability that general instruction datasets lack.

- **Hardcoded**: 140 examples manually written by the authors with prompts like "Tell me about yourself," designed so the model correctly states its name and developer identity. This is a targeted intervention to ensure the model has accurate self-knowledge and does not hallucinate that it is GPT-4 or another system.

**Datasets removed from V1:**

- **Dolly** (Databricks, 2023): Explicitly removed "due to its poor performance in previous ablations." Dolly was a human-written instruction dataset of ~15,000 examples, but Wang et al. (2023b) had found it contributed little to overall performance.

**Filtering decisions:**

The paper applies a specific filter: "any samples that include references to other LLM systems such as GPT-4, Open Assistant, or Claude" are removed. The stated reason is "to avoid contradicting the hardcoded prompts." Without this filter, the model would see training examples where it refers to itself as "GPT-4" (in ShareGPT data, for instance) and also training examples where the hardcoded responses identify it as "TÜLU." This would create a contradiction in the training data, potentially causing the model to produce inconsistent self-identification at test time.

**Resulting mixture statistics:**

After all construction steps, TÜLU-V2-mix contains **326,154 examples**, compared to 490,445 in the V1 mixture — a **33% reduction in dataset size** despite adding several new datasets. This is achieved through aggressive downsampling of large sources (FLAN is reduced to 100,000 total examples including CoT, down from a larger fraction of V1) and the removal of Dolly. The paper's philosophy is that quality and diversity matter more than quantity, and the performance results (Table 2: V2 outperforms V1 by 8% on average at 7B) validate this choice.

**Length statistics (Figure 1):**

- Mean length: 1,097 tokens
- 25th percentile: 230 tokens
- 75th percentile: 1,464 tokens

The distribution has a long tail — some examples (particularly from ShareGPT and Open Assistant conversations) extend to several thousand tokens. This motivates the decision to increase the maximum training sequence length from 2,048 (used in TÜLU 1) to 8,192 tokens: at 2,048, 63,900 examples (roughly 20% of the dataset) would be truncated. At 8,192, only 20 examples are truncated, "better capturing the long tail of lengthy examples in our training data."

**Design choice: why use a mixture rather than a single best dataset?**

The paper inherits and extends the central finding from TÜLU 1: no single dataset is optimal for all evaluation dimensions. Table 2 shows this concretely: a model trained on ShareGPT alone achieves high AlpacaEval (72.3% at 7B) but poor MMLU (47.8%) and GSM8k (20.0%), while the V2 mixture balances these tradeoffs. The V2 mixture is constructed to provide complementary coverage: FLAN and CoT provide task-oriented reasoning structure, ShareGPT and WizardLM provide conversational fluency and instruction-following complexity, Code-Alpaca maintains programming ability, Science Literature adds domain expertise, and hardcoded examples ensure consistent self-identification. The hypothesis — validated by the overall results — is that this complementary coverage produces a model that is strong across all dimensions rather than excelling in one at the expense of others.

**Design choice: why downsample rather than use all data?**

The paper does not explicitly justify the specific downsampling ratios, but the implicit logic is that larger datasets (FLAN, ShareGPT) would dominate the training distribution if used in full, causing the model to overfit to their specific patterns at the expense of smaller but high-quality sources like LIMA or the science literature mixture. By downsampling to roughly equal proportions, the V2 mixture ensures each dataset meaningfully contributes to the final model. The total size of 326,154 examples, trained for 2 epochs, means the model sees approximately 652,308 training examples — a substantial but not enormous fine-tuning budget that requires careful curation of what is included.

---

#### Supervised Fine-Tuning (SFT) Stage

The SFT stage takes a pretrained base model (LLAMA-2 or CODE LLAMA) and fine-tunes it on TÜLU-V2-mix using a standard language modeling objective: predict the next token in the response given the instruction and any previous conversation turns. This is identical in principle to the original TÜLU fine-tuning, but with updated base models, data mixture, and hyperparameters.

**Base models:**

The paper uses two families:

- **LLAMA-2** (Touvron et al., 2023b): Decoder-only transformer models pretrained on 2 trillion tokens (roughly twice LLAMA-1's 1–1.4 trillion). Available at 7B, 13B, and 70B parameters. The paper states LLAMA-2 shows "a 10% average improvement across model sizes on a set of academic benchmarks" compared to LLAMA-1, making it a substantially stronger starting point. The architecture is largely identical to LLAMA-1 — rotary position embeddings, SwiGLU activations, RMSNorm — meaning improvements come from data scale and quality rather than architectural innovation.

- **CODE LLAMA** (Roziere et al., 2023): LLAMA-2 models further pretrained on code data (the paper does not specify the exact code pretraining corpus size). Available at 7B, 13B, and 34B parameters (no 70B version). The continued code pretraining means these models have different weight distributions than LLAMA-2 — they are specialized for code but the paper investigates how this specialization affects general instruction-following.

**Training hyperparameters:**

The paper uses a consistent set of hyperparameters across all model sizes, with one exception (learning rate for 70B):

- Precision: BFloat16 (half-precision floating point with a larger dynamic range than FP16, reducing memory usage while maintaining training stability)
- Epochs: 2
- Weight decay: 0 (no L2 regularization — common in LLM fine-tuning since the pretrained weights already serve as an implicit regularizer)
- Warmup ratio: 0.03 (linear learning rate warmup over the first 3% of training steps to avoid destabilizing the pretrained weights with large initial gradients)
- Learning rate: `$2 \times 10^{-5}$` for 7B and 13B, reduced to `$1 \times 10^{-5}$` for 70B
- Maximum sequence length: 8,192 tokens (quadrupled from TÜLU 1's 2,048)
- Effective batch size: 128 (number of examples per gradient update, accumulated across multiple micro-batches if necessary to fit in memory)

**Why reduce learning rate for 70B?** Larger models have more parameters, making them more sensitive to training instability — a given per-parameter gradient update can push the model further from its pretrained distribution. Reducing the learning rate for the largest model is a common practice to maintain stable training, and the paper follows it without further ablation.

**Why 2 epochs?** The paper does not justify this choice explicitly, but it likely reflects a balance between sufficient training to learn the instruction-following behavior and avoiding overfitting. With 326,154 examples and 2 epochs, the model sees approximately 652,308 training examples. More epochs risk the model memorizing specific instruction-response pairs rather than learning generalizable instruction-following, especially given the relatively small dataset size compared to pretraining corpora.

**Why 8,192 context length?** As discussed in the data construction section, this is driven by the length distribution of TÜLU-V2-mix. At 2,048, roughly 20% of examples would be truncated, losing important training signal from long conversations (ShareGPT, Open Assistant) and long-form reasoning traces (CoT, WizardLM). At 8,192, only 20 examples are truncated. However, training with longer sequences substantially increases memory requirements — the attention computation scales quadratically with sequence length. This tradeoff is feasible at SFT time because fine-tuning is much cheaper than pretraining, but it does mean the SFT stage requires more compute than it would with a shorter context.

**Training infrastructure:**

All full-model fine-tuning (excluding QLoRA experiments) was performed on TPU v3 pods:
- 256-chip pods for most models
- 512-chip pods for the 70B DPO training (the largest model + longest training combination)

The codebase is based on EasyLM (Geng, 2023), a JAX-based training framework designed for large language models. The paper also releases an alternative PyTorch-based fine-tuning codebase at `https://github.com/allenai/open-instruct`.

**What the SFT objective actually does:**

The training objective is standard autoregressive language modeling — the model is trained to maximize the probability of the response tokens given the instruction tokens. Importantly, the loss is computed only on response tokens, not on instruction tokens. This means the model learns to produce appropriate responses but does not update its representation of the instruction format itself (which it inherits from the base model's pretraining).

The V2 mixture uses a chat template to format examples consistently. For a single-turn instruction-response pair, the template structures the input as:

```
<|user|>
{instruction}
<|assistant|>
{response}
```

The model is trained to predict the response tokens after the `<|assistant|>` marker, conditioned on all preceding tokens. For multi-turn conversations (ShareGPT, Open Assistant), the template alternates between `<|user|>` and `<|assistant|>` markers, with the model trained to predict each assistant turn.

**Design choice: why full fine-tuning as the default?**

The paper treats full fine-tuning as the default and studies QLoRA as a cost-reduction alternative rather than the reverse. This reflects a prioritization of performance over accessibility: full fine-tuning produces the strongest models, and the paper wants to establish the upper bound before exploring approximations. This differs from some other open-source projects (e.g., Alpaca, which used LoRA by default) and positions TÜLU 2 as targeting the highest possible quality within the open-weight paradigm.

---

#### Direct Preference Optimization (DPO) Stage

DPO is the most technically sophisticated component of the TÜLU 2 pipeline. It takes an already instruction-tuned model (the SFT model) and further optimizes it to produce outputs that align with human (or AI-as-judge) preferences, without requiring a separate reward model or online sampling. The paper follows the Zephyr-Beta recipe (Tunstall et al., 2023) closely but demonstrates — for the first time — that the approach scales stably to 70B parameters.

**What DPO optimizes:**

DPO (Rafailov et al., 2023) is built on a theoretical insight: the optimal policy for an RLHF objective (maximizing reward while staying close to a reference policy) can be expressed in closed form as a function of the reward model. This means the policy itself implicitly represents a reward function — "your language model is secretly a reward model," as the DPO paper's title declares. DPO exploits this by directly optimizing the preference prediction objective:

The standard RLHF objective (which DPO was designed to simplify) is:

$$\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi(y|x)} [r(x, y)] - \beta \cdot D_{KL}[\pi(y|x) \| \pi_{\text{ref}}(y|x)]$$

where `$\pi$` is the policy being trained, `$\pi_{\text{ref}}$` is a reference policy (typically the SFT model), `$r(x, y)$` is the reward for output `$y`$ given input `$x$`, `$D_{KL}$` is the Kullback-Leibler divergence measuring how far `$\pi$` diverges from `$\pi_{\text{ref}}$`, and `$\beta$` is a hyperparameter controlling the tradeoff between reward maximization and staying close to the reference policy.

**What this objective means:** it says "find a policy that gets high reward, but don't stray too far from the original SFT model's behavior." The KL penalty term prevents the policy from exploiting the reward model by producing outputs that score highly but are unnatural.

DPO derives a loss function from this objective that does not require actually training a reward model or running the RL optimization. Instead, given a dataset of preference pairs `$(x, y_w, y_l)$` where `$y_w$` is the preferred response and `$y_l$` is the dispreferred response for prompt `$x$`, the DPO loss is:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[\log \sigma \left( \beta \cdot \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \cdot \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right)\right]$$

where `$\pi_\theta$` is the policy being optimized (initialized from the SFT model), `$\pi_{\text{ref}}$` is the frozen reference policy (the SFT model), `$\beta$` is the same KL penalty coefficient from the RLHF objective, `$y_w$` is the preferred ("winning") response, `$y_l$` is the dispreferred ("losing") response, `$\sigma$` is the logistic sigmoid function, and `$D$` is the preference dataset.

**What it computes in operational terms:** for each preference pair, DPO computes the ratio of the policy's probability of the winning response to the reference policy's probability of the winning response (call this `$r_w$`), and the same ratio for the losing response (call this `$r_l$`). It then computes the difference `$\beta \cdot (\log r_w - \log r_l)$`, passes this through the sigmoid, and minimizes the negative log-likelihood. This means the loss is low when the policy assigns higher relative probability to winning responses compared to the reference policy, and lower relative probability to losing responses. The gradient pushes `$\pi_\theta$` to increase `$\pi_\theta(y_w | x)$` and decrease `$\pi_\theta(y_l | x)$`, with the reference model anchoring the scale — the policy cannot simply increase probability of everything with high reward, because the ratio to the reference model must change.

**Why this form:** the key insight is that under the Bradley-Terry model of preferences (which assumes the probability that `$y_w$` is preferred over `$y_l$` depends on the difference in their true rewards), the optimal policy `$\pi^*$` satisfies `$r(x,y) = \beta \cdot \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \text{constant}$`. Plugging this into the Bradley-Terry preference probability and maximizing likelihood yields the DPO loss. The alternative — PPO-based RLHF — requires training a separate reward model, then using it to score online samples from the policy during RL training, which involves running multiple models simultaneously, managing reward model overfitting, and tuning PPO hyperparameters. DPO replaces all of this with a simple supervised loss on preference pairs, making it dramatically simpler to implement and tune.

**The preference dataset: UltraFeedback**

The paper uses UltraFeedback (Cui et al., 2023), the same dataset used by Zephyr-Beta, rather than collecting new preferences. UltraFeedback contains approximately 64,000 prompts, each with 4 responses generated by different models (including GPT-4, GPT-3.5, and several open models). GPT-4 then rates these responses on dimensions like helpfulness, honesty, and instruction-following, producing a ranking. The Zephyr-Beta recipe "binarizes" this data: for each prompt, the highest-rated response is treated as `$y_w$` and a randomly selected lower-rated response as `$y_l$`, creating a binary preference pair. The paper does not specify filtering criteria beyond "filtered and binarized form," following Zephyr-Beta.

**DPO training hyperparameters:**

The paper uses hyperparameters directly from Zephyr-Beta, which is notable because Zephyr-Beta only tested at 7B:

- Precision: BFloat16
- Epochs: 3
- Weight decay: 0
- Warmup ratio: 0.1
- Learning rate: `$5 \times 10^{-7}$` (this is the critical hyperparameter — see below)
- Maximum sequence length: 8,192
- Effective batch size: 32
- Beta (`$\beta$` in the DPO loss): 0.1

**Why the learning rate is so low:**

The learning rate for DPO (`$5 \times 10^{-7}$`) is **40 times lower** than the 70B SFT learning rate (`$1 \times 10^{-5}$`) and **400 times lower** than the 7B/13B SFT learning rate (`$2 \times 10^{-5}$`). The paper states this low learning rate is "required for stable and effective DPO training." This is not explained theoretically but has an intuitive justification: the SFT model already produces reasonable outputs, and DPO is making relatively subtle adjustments to relative probabilities of responses. A high learning rate would cause the policy to diverge too far from the reference model, violating the implicit assumption behind DPO that the reference model provides a meaningful anchor. The KL penalty in the DPO objective is theoretical — it emerges from the derivation — but in practice, large gradient steps can cause the policy to move into regions where the ratio `$\pi_\theta(y|x) / \pi_{\text{ref}}(y|x)$` becomes extreme, leading to training instability. The low learning rate keeps updates small enough that the reference ratio remains well-behaved.

The paper also notes that DPO training the 70B model took approximately 7 days on a 512-core TPU v3 pod, compared to shorter training for smaller models, reflecting both the larger model size and the 3-epoch training duration.

**What DPO actually changes in the model:**

The results in Table 3 and Table 4 show a clear pattern: DPO dramatically improves open-ended generation quality (AlpacaEval improves by 8-12 percentage points across scales; MT-Bench improves substantially for 13B and 70B) while leaving most capability-focused metrics essentially unchanged (MMLU moves by ±0.5 points; GSM8k by ±1-3 points; BBH by 0-3 points). This suggests DPO is primarily shaping the *style and quality* of model outputs rather than the underlying factual or reasoning capabilities — the model becomes better at producing responses that GPT-4-as-judge prefers, without becoming more knowledgeable or capable at math.

Two consistent side effects emerge:

1. **Multilingual capability degrades**: TydiQA scores drop substantially after DPO — from 46.4 to 44.5 (7B), from 53.2 to 39.7 (13B), and from 53.6 to 35.8 (70B). The 13B drop of 13.5 points and the 70B drop of 17.8 points are the largest single negative effects of DPO. The paper hypothesizes this is because "both our supervised finetuning and DPO data mixes do not explicitly contain multilingual data, and are majority English-language," so DPO training makes multilingual outputs "further out-of-distribution." This is a concrete warning: DPO can amplify the language distribution of the preference data, and if that data is English-only, multilingual performance suffers.

2. **Verbosity increases**: Table 4 shows that the average output length of TÜLU 2+DPO models on AlpacaEval is consistently higher than TÜLU 2 without DPO (e.g., 1,437 vs. 1,248 tokens at 7B; 1,414 vs. 1,011 at 70B). This aligns with prior work showing that RLHF training biases models toward longer responses, likely because human raters (and GPT-4 as a rater) tend to perceive more detailed responses as more helpful. However, the paper notes that TÜLU 2+DPO models are "dramatically less verbose than other open-weight models" — for comparison, Zephyr-Beta 7B averages 2,721 tokens, nearly twice TÜLU 2+DPO 7B's 1,437 tokens.

**Design choice: why DPO and not PPO?**

The paper states this explicitly: "we use the direct preference optimization (DPO) algorithm due to the simplicity of its implementation." PPO-based RLHF requires training a separate reward model, managing online sampling during training, balancing multiple loss terms (policy loss, value loss, entropy bonus), and dealing with the engineering complexity of running multiple models simultaneously. DPO reduces this to a single supervised loss on a static dataset, using the same training infrastructure as SFT. For a research project focused on evaluating recipe combinations, this simplicity is a major advantage — it allows the paper to explore DPO at three model scales (including the unprecedented 70B) without the engineering burden that PPO at 70B would require.

**Stability at 70B: a key empirical finding:**

The paper states "we find these hyperparameters scale, providing stable training and performance improvements for models at all sizes." This is a non-trivial finding because DPO had not been demonstrated at 70B before. Training instability at large scales could manifest as loss spikes, policy collapse (where the model outputs degenerate or repetitive text), or overfitting to the preference data. The fact that the 7B-optimized hyperparameters transfer directly to 70B — same learning rate, same beta, same number of epochs — suggests that DPO's optimization dynamics are well-behaved across scale, which is important for future work that wants to apply DPO to even larger models.

---

#### QLoRA Parameter-Efficient Alternative

The paper explores QLoRA (Dettmers et al., 2023) as a drop-in replacement for full fine-tuning during the SFT stage, motivated by the goal of reducing computational requirements. QLoRA combines two techniques: 4-bit NormalFloat quantization of the base model weights (reducing memory usage by ~4x compared to 16-bit) and low-rank adaptation (LoRA) which freezes the quantized base weights and trains small rank-decomposition matrices that are added to the frozen weights.

**QLoRA hyperparameters:**

The paper uses the following settings, which it states were selected through "a variety of QLoRA hyperparameters" experiments at smaller scale:

- Epochs: 5 (more than full fine-tuning's 2, likely because LoRA updates are lower-capacity per step)
- Weight decay: 0
- Warmup ratio: 0.03
- Learning rate: `$1 \times 10^{-4}$` (5-10x higher than full fine-tuning, compensating for the smaller effective parameter count)
- Maximum sequence length: 4,096 (half the full fine-tuning's 8,192 — the paper does not explain this reduction, but it may be due to memory constraints or LoRA-specific optimization)
- Effective batch size: 128
- LoRA rank: 64 (the dimensionality of the low-rank approximation matrices)
- LoRA alpha: 16 (a scaling factor for the LoRA updates; the effective update is `$\frac{\alpha}{r} \cdot \Delta W$`)
- LoRA dropout: 0.1
- Layers wrapped: all attention and feedforward linear layers (the broadest possible application of LoRA, maximizing the capacity of the adaptation)

**How QLoRA works in this context:**

The pretrained LLAMA-2 weights are quantized to 4-bit precision using NormalFloat4, a quantization scheme designed to be optimal for normally-distributed weights (which LLM weights approximately are). These quantized weights are frozen — no gradients are computed for them. For each linear layer specified (all attention query, key, value, and output projections, plus all feedforward layers), two low-rank matrices `$A \in \mathbb{R}^{d_{\text{in}} \times r}$` and `$B \in \mathbb{R}^{r \times d_{\text{out}}}$` are added, where `$r = 64$` is the rank. The forward pass computes `$y = W_{\text{quantized}} \cdot x + \frac{\alpha}{r} \cdot B \cdot A \cdot x$`. Only `$A$` and `$B$` are trained.

The total trainable parameters are `$r \times (d_{\text{in}} + d_{\text{out}})$` per layer, which for a 70B model with hidden dimension 8,192 would be `$64 \times (8192 + 8192) = 1,048,576$` parameters per layer, times the number of wrapped layers. The total is a small fraction of the 70B full parameters, making training feasible on a single A100 80GB GPU (the paper used an internal A100 cluster).

**Results (Table 5):**

The key finding is that QLoRA "does not match full fine-tuning in long-form generation tasks." At 7B, AlpacaEval drops from 73.9% to 56.1% — a 20% relative decline. At 70B, the gap narrows to 86.6% vs. 78.6% — roughly an 8% relative decline. On capability benchmarks like MMLU, QLoRA is much closer: 50.4 vs. 48.8 at 7B, 67.3 vs. 67.4 at 70B. The gap shrinks with model size (average gap of 10% at 7B, 3% at 70B), consistent with prior work on parameter-efficient tuning (Lester et al., 2021).

The paper hypothesizes that the discrepancy in AlpacaEval performance — which persists even at 70B — may be because Dettmers et al. (2023) primarily validated QLoRA on MMLU, which this paper also finds shows small gaps. The broader evaluation suite reveals dimensions (open-ended generation) where the approximation error of LoRA matters more. This makes intuitive sense: open-ended generation requires the model to produce coherent, stylistically diverse long-form text, which may depend on fine-grained weight interactions that low-rank approximations cannot fully capture. MMLU, by contrast, requires selecting the correct multiple-choice answer, which may be more robust to approximation.

**Why QLoRA was only tested at SFT, not DPO:**

The paper states: "Due to sub-par performance at the instruction tuning stage, we did not explore using QLoRA during RLHF training." If the SFT model produced by QLoRA is already weaker than the full fine-tuning baseline, applying DPO on top would not allow a clean comparison — the SFT starting points would be different, making it unclear whether differences in final performance came from QLoRA's SFT approximation or its interaction with DPO.

---

#### Evaluation Framework

The evaluation suite is the lens through which all model variants are compared. The paper reuses and extends the framework from TÜLU 1, with two changes: replacing AlpacaFarm with AlpacaEval for open-ended generation evaluation, and adding MT-Bench as an additional open-ended generation benchmark.

**Capability benchmarks (structured tasks):**

- **MMLU** (Massive Multitask Language Understanding): 57 subjects spanning STEM, humanities, social sciences. Models answer multiple-choice questions with 0 few-shot examples (no in-context examples provided). Reported as average accuracy across all subjects. This tests broad factual knowledge and reasoning.

- **GSM8k**: Grade-school math word problems requiring multi-step arithmetic reasoning. Models are prompted with 8 few-shot chain-of-thought examples followed by the test question. Answers are extracted by taking the last number in the model's response (since all answers are numeric). Reported as average accuracy. This tests mathematical reasoning specifically.

- **BBH** (Big Bench Hard): A subset of 23 challenging tasks from BIG-Bench, selected because they were beyond the capabilities of earlier models. Models are prompted with 3 few-shot chain-of-thought examples. For CoT tasks, the answer is extracted as "the first word after the phrase 'So the answer is'" or the entire response if that phrase is absent. Reported as average accuracy over sub-tasks. This tests complex reasoning across diverse domains.

- **TydiQA** (Gold Passage setting): Multilingual question answering where the model is given a passage in one of 11 languages and must answer a question about it. Only the GoldP (Gold Passage) setting is used, where the correct passage is provided. One in-context example is used. Reported as average F1 score. This tests multilingual comprehension.

- **CodexEval** (HumanEval): 164 Python programming problems where the model is given a function signature and docstring and must complete the function body. Following the original Codex paper (Chen et al., 2021), unbiased estimates of pass@k are computed by sampling multiple completions (temperature 0.8) and checking functional correctness against hidden test cases. pass@10 is reported. This tests code generation capability.

- **TruthfulQA**: 818 questions designed to probe whether models reproduce common human misconceptions. Models are prompted with 6 few-shot QA examples and greedy decoding. Two GPT-based classifiers are trained to judge truthfulness and informativeness of responses. The paper reports "% Informative and Truthful" — the fraction of responses that are both truthful and informative. This tests whether the model generates factually correct answers rather than plausible-sounding falsehoods.

**Safety evaluation:**

- **ToxiGen**: A dataset of prompts designed to elicit toxic language about specific demographic groups. The paper uses only "hateful" prompts and 500 prompts per group to reduce evaluation cost. For instruction-tuned models, the prompt is placed in the chat template and the model generates until a stop token or 512 tokens. Generated text is passed through a RoBERTa-large toxicity classifier finetuned by Hartvigsen et al. (2022). The metric is the percentage of generations classified as toxic (lower is better).

**Open-ended generation benchmarks (GPT-4 as judge):**

- **AlpacaEval**: 805 prompts from diverse sources. The model generates responses (up to 8,192 tokens), and GPT-4-0613 compares each response to a reference response from Davinci-003 (a GPT-3 variant), declaring a winner. The reported metric is the win rate — the percentage of prompts where the model's response is preferred over Davinci-003's. The paper uses "alpaca_eval_gpt4" as the annotator configuration. The paper notes that AlpacaEval "does not use a pinned GPT-4 version," so they ensure all evaluations use GPT-4-0613 for consistency.

- **MT-Bench**: 80 multi-turn questions across 8 categories (STEM, Humanities, Reasoning, Coding, Math, Extraction, Roleplay, Writing), each with a follow-up question, for 160 total responses graded by GPT-4-0613 on a 1-10 scale. The paper uses single-answer grading (each response is scored independently, not as a pairwise comparison). The average score across all 160 responses is the primary metric. The full breakdown by category is provided in Appendix D (Table 8).

**Contamination issues:**

The paper explicitly flags that TruthfulQA prompts appear in UltraFeedback, the dataset used for DPO training. As a result, "we omit TruthfulQA results when showing comparisons with contaminated models (any models trained with the UltraFeedback dataset)." This is a careful methodological choice — DPO-trained models might have memorized TruthfulQA answers from their training data, making their TruthfulQA scores unreliable indicators of genuine truthfulness. This contamination does not affect SFT-only models (TÜLU 2 without DPO), which can still be evaluated on TruthfulQA.

The paper also notes that they "cannot rule out the possibility" that GPT-4 and GPT-3.5 models "are trained on the evaluation benchmark datasets," which would inflate their scores relative to models that have not seen the benchmarks during training. This is a standard caveat in the field.

**How models are compared:**

The evaluation framework enables direct comparisons because all models are evaluated on the same benchmarks with the same prompts and the same GPT-4 versions. When the paper claims "TÜLU 2 70B achieves similar performance to GPT-3.5-turbo-0301 in MMLU, BBH and TydiQA," this is based on the numbers in Table 1: MMLU 67.3 vs. 67.9, BBH 68.4 vs. 66.1, TydiQA 53.6 vs. 51.9. The comparisons are valid because the evaluation protocol is identical for both models.

**Average score computation:**

The paper reports an "Average" column in most tables, computed as a naive average across tasks. For ToxiGen, the value averaged is `$100 - \text{% Toxic}$` so that higher is better (consistent with all other metrics). For TruthfulQA, only the "% Info+True" metric is included in the average. This averaging is acknowledged as "naive" — it weights all benchmarks equally despite differences in difficulty, reliability, and domain coverage — but provides a single scalar for quick comparison.

## 4. Key Insights and Innovations

### Innovation 1: The Recipe Integration Paradigm — Treating Instruction Tuning as a Compositional System

The TÜLU 1 and TÜLU 2 papers together establish something subtler than a single model release: they demonstrate that **open instruction tuning is best understood not as a monolithic algorithm, but as a modular system where base models, data mixtures, and post-training stages compose in predictable, evaluable ways.** This is a conceptual reframing rather than a technical novelty — the individual components (LLAMA-2, V2-mix, DPO) all existed before this paper — but the contribution lies in treating the *integration itself* as the object of scientific study.

Before TÜLU 2, the dominant pattern in open instruction tuning was point-solution releases: a single team would pick one base model (e.g., MISTRAL), one data mixture (e.g., UltraChat), and one post-training method (e.g., DPO), report final numbers against existing benchmarks, and release the model. Zephyr-Beta (Tunstall et al., 2023) and Xwin-LM (Xwin-LM Team, 2023) followed this pattern. Each release established a new state-of-the-art point, but the field had no systematic way to answer counterfactual questions: *would DPO have helped as much if the SFT data were different? Would QLoRA have performed better on a different base model? Does the optimal data mixture change with model scale?*

TÜLU 2 addresses this by constructing a sparse factorial experiment: fix the evaluation suite, then independently vary (a) base model (LLAMA-2 vs. CODE LLAMA), (b) data mixture (V1 vs. V2 vs. ShareGPT-only), (c) fine-tuning method (full vs. QLoRA), and (d) DPO stage (present vs. absent). The results are reported as composable deltas: "V2 mix outperforms V1 by 8% at 7B, but only 1% at 70B" (Table 2); "DPO improves AlpacaEval by ~10% regardless of scale, but degrades TydiQA by ~15 points at 70B" (Table 3); "CODE LLAMA improves CodexEval by ~30 points but costs ~20 points on AlpacaEval" (Table 6). These deltas are not individually novel — many were hinted at in prior work — but **their simultaneous measurement on a fixed evaluation scaffold** converts scattered claims into a coherent picture of tradeoffs.

This matters beyond TÜLU 2 itself because it establishes a template for how the open-source community can make cumulative progress. When a new base model (say, MISTRAL-2) or a new post-training method (say, KTO) is released, rather than building a bespoke pipeline from scratch, future work can drop the new component into the TÜLU framework, rerun evaluation, and report the delta. The paper makes this concrete by releasing all code, data, and checkpoints. This is a fundamentally different scientific strategy than the point-solution approach: it prioritizes **composability and comparability** over raw state-of-the-art number chasing.

The limitation, of course, is that the factorial design is sparse — only a handful of combinations are tested, and interactions cannot be estimated without full crossing (e.g., we don't know how QLoRA interacts with DPO because it was never tested). But the conceptual architecture is what matters. The paper establishes that **a well-factored evaluation scaffold enables the field to reason about instruction-tuning components as interchangeable parts**, which is a meta-scientific contribution that outlives any specific model release.

---

### Innovation 2: DPO at 70B — Stability as an Empirical Discovery, Not a Guarantee

The paper's demonstration that DPO training scales stably to 70 billion parameters — producing TÜLU 2+DPO 70B, the largest publicly released DPO model at the time — is significant not because the result was theoretically surprising, but because **the absence of evidence at this scale was a genuine barrier to adoption.** Before TÜLU 2, DPO had only been demonstrated at 7B (Zephyr-Beta on MISTRAL) and informally at smaller scales. The theoretical properties of DPO — that it is equivalent to RLHF under the Bradley-Terry preference model — do not guarantee that the loss landscape at 70B will be as well-behaved as at 7B. Larger models have sharper minima, more complex loss surfaces, and different sensitivity to the KL constraint that DPO implicitly enforces.

The concern is not hypothetical. The paper reports that a learning rate of 5 × 10⁻⁷ was "required for stable and effective DPO training" — this is 400× lower than the SFT learning rate for 7B/13B and 20× lower than the 70B SFT rate. At higher learning rates, DPO training can diverge because the policy moves too far from the reference model, violating the local approximation that makes DPO equivalent to RLHF. The fact that the *same* hyperparameters used for 7B Zephyr-Beta — learning rate, beta, epochs, batch size — transferred directly to 70B without modification is an empirical discovery: the DPO loss curvature does not change dramatically with scale in the LLAMA-2 architecture.

Why does this matter for the field? RLHF at 70B+ using PPO is an engineering nightmare requiring multiple models in memory, online sampling, reward model training, and careful hyperparameter tuning to prevent reward hacking. The largest publicly documented PPO-trained models at the time were LLAMA-2-Chat 70B (Meta, using proprietary infrastructure) and a handful of smaller open efforts. DPO reduces the entire preference optimization to a single supervised training run on static data, using the same infrastructure as SFT. The paper's demonstration that this works at 70B — "completing three epochs in approximately 7 days" on a 512-core TPUv3 — provides a concrete cost baseline that makes DPO dramatically more accessible than PPO for teams that cannot afford the reinforcement learning infrastructure.

The significance is amplified by the result pattern: DPO improves AlpacaEval by 8-12 points across all scales while leaving capability benchmarks essentially unchanged (Table 3). This suggests that DPO at 70B is not merely "possible" but **produces qualitatively the same benefits as at smaller scales**, with no evidence of diminishing returns or new failure modes. The paper can thus claim — and does — that DPO "is a promising path for training large models on human feedback without the engineering complexity required by PPO."

The negative result on multilingual performance (TydiQA drops ~17 points at 70B) is equally informative: it reveals that DPO's scaling behavior includes an amplification of the training data's language distribution, and that this amplification gets *worse* with scale (7B drop: -1.9; 13B drop: -13.5; 70B drop: -17.8). This is a diagnostic finding: DPO does not just preserve the SFT model's capabilities, it sharpens them along the dimensions represented in the preference data, and if that data is English-only, multilingual capability erodes proportionally to model size. This finding was not predictable from the DPO theory alone and represents an empirical discovery about how preference optimization interacts with scale.

---

### Innovation 3: The Long-Form Generation Gap — QLoRA's Capability Profile as a Diagnostic Tool

The paper's finding that QLoRA produces an ~20% gap on AlpacaEval relative to full fine-tuning, while matching on MMLU within ~1 point, is not just a cost-performance tradeoff result. It is a **diagnostic insight about what aspects of model capability depend on high-fidelity weight updates versus what can be captured by low-rank approximations.** This reframes QLoRA from a generic "efficient fine-tuning method" into a lens for understanding the structure of instruction-following.

Prior work (Dettmers et al., 2023) had validated QLoRA primarily on MMLU, where it matched full fine-tuning performance closely. This paper's contribution is to show that MMLU is the *wrong metric* for assessing QLoRA's practical impact on instruction-tuned models. The benchmarks that matter most for deployed instruction-following systems — open-ended generation quality, as measured by AlpacaEval and (for the 7B model) MT-Bench — are precisely where QLoRA underperforms most severely. The paper identifies this as a systematic pattern: QLoRA's approximation error is not uniform across task types. It hits hardest on tasks requiring the model to generate coherent, stylistically diverse long-form text, and is least damaging on tasks requiring factual recall or structured reasoning.

Why is this conceptually significant? It suggests that there is a **qualitative difference in the weight-space geometry** of the capabilities that SFT imparts. Factual knowledge and multiple-choice reasoning — the skills tested by MMLU — may be encoded in the base model's weights in ways that can be oriented or redirected by low-rank updates: a rank-64 adapter can tilt the model toward selecting correct answers without needing to modify fine-grained token-level generation patterns. But open-ended generation — producing the right tone, length, structure, and style for a diverse set of prompts — may require **higher-rank modifications** that restructure the interactions between attention heads and feedforward layers in ways that a rank-64 bottleneck cannot capture.

The paper does not prove this mechanism, but the pattern itself is the diagnostic contribution. The gap shrinks with model size (20% at 7B → ~8% at 70B), which further suggests that larger models have more redundancy in their representations, allowing lower-rank adaptations to capture more of the needed modifications. This is consistent with prior work on parameter-efficient tuning (Lester et al., 2021) but the paper extends it to a broader task suite and identifies the specific task type — open-ended generation — where the gap persists longest.

This finding has direct practical implications: for projects where the primary evaluation is MMLU (academic benchmarks, factual QA systems), QLoRA is a very cost-effective choice. For projects where open-ended generation quality matters (chatbots, assistants, creative writing tools), QLoRA's savings come at a substantial capability cost that does not fully disappear even at 70B. The paper thus provides a **decision heuristic** rather than a blanket recommendation, which is more useful to practitioners than a simple "QLoRA works" or "QLoRA doesn't work" claim.

---

### Innovation 4: CODE LLAMA as a Capability Tradeoff — Domain Specialization Is Not Additive

The CODE TÜLU 2 results (Table 6) establish something that the original CODE LLAMA paper (Roziere et al., 2023) did not systematically investigate: **continued pretraining on specialized data reshapes model capabilities in non-target domains in ways that are large, consistent, and not trivially recoverable through instruction tuning.** This is a finding about the structure of language model pretraining rather than about any specific fine-tuning recipe.

The numbers are striking. At 7B, CODE TÜLU 2 achieves 68.9% on CodexEval compared to TÜLU 2's 36.9% — a 32-point improvement, roughly doubling coding capability. But on AlpacaEval, CODE TÜLU 2 scores 58.0% compared to TÜLU 2's 73.9% — a roughly 16-point decline. At 13B, the pattern is similar: +27 points on CodexEval (76.2 vs. 49.0), -15 points on AlpacaEval (64.1 vs. 78.9). The paper summarizes this as a "20% average drop in performance" on open-ended generation.

This is not a trivial "specialization means tradeoffs" observation. The magnitude matters: a 16-20 percentage point drop on the primary open-ended generation benchmark is a substantial capability regression, making CODE TÜLU 2 worse at general instruction-following than the non-code-specialized TÜLU 2, despite being trained on the same instruction data. The paper explicitly notes that CODE LLAMA-Instruct (Meta's own instruction-tuned variant) outperforms CODE TÜLU 2 on AlpacaEval (75.3 vs. 64.1 at 13B), suggesting that Meta's internal instruction-tuning recipe may have specifically targeted this regression.

The conceptual insight is that **continued pretraining on code does not simply add coding capability to the existing LLAMA-2 capability profile; it partially overwrites it.** The code pretraining shifts the model's internal representations toward patterns that are useful for code generation (likely: more structured attention patterns, different token-level statistics, different sensitivity to syntactic constraints) and away from patterns that support the stylistic flexibility needed for diverse open-ended instruction-following. The V2 instruction-tuning mixture, which was designed for LLAMA-2 base models, partially recovers general capability — CODE TÜLU 2 does outperform CODE LLAMA-Instruct on average — but cannot fully reverse the representational shift induced by the code pretraining.

This finding has implications for the broader trend toward domain-specialized base models. If continued pretraining on math, or medicine, or law similarly reshapes capabilities in non-obvious ways, then the field needs evaluation suites that test *regression* in non-target domains, not just improvement in the target domain. The original CODE LLAMA paper focused overwhelmingly on code benchmarks; TÜLU 2's evaluation suite reveals the cross-domain cost. The paper thus provides a methodological lesson: **domain specialization should be evaluated on a broad capability battery, not just on the target domain, because capability tradeoffs can be large and asymmetric.**

The result also complicates the narrative that domain-specific models can simply be smaller and cheaper than general models. At 7B, CODE TÜLU 2 matches TÜLU 2+DPO 70B on CodexEval (68.9 vs. 68.9) — a 10× reduction in model size for coding tasks specifically. But if the deployment requires both coding and general instruction-following, CODE TÜLU 2's regression on AlpacaEval may be unacceptable, forcing a choice between running two specialized models (one for code, one for general use) or accepting the larger general-purpose model. The paper makes this tradeoff explicit and quantifiable.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates primarily on the MATH benchmark (Hendrycks et al., 2021), using the specific split from Lightman et al. (2022): 12,000 training questions and 500 test questions. MATH consists of high-school competition-level mathematics problems spanning algebra, geometry, probability, and number theory. The paper chooses MATH because test-time compute is expected to help most when the model already possesses the necessary knowledge and the challenge is drawing complex inferences — mathematical reasoning fits this profile since it requires multi-step logical deduction rather than novel factual recall. The training set is used for generating PRM training data and revision model training data; the test set is used for all reported evaluations.

- **Base model(s).** All experiments use PaLM 2-S* (Codey), a model the authors describe as "representative of the capabilities of many contemporary LLMs." The pass@1 on MATH is roughly 10–19% depending on the prompt and sampling configuration — non-trivial but far from saturation, leaving substantial room for test-time compute to improve performance. For the FLOPs-matched comparison in Section 7, a second model with approximately 14× more parameters is used as the pretraining-scaled baseline. The paper fixes training data and scales only model parameters when increasing pretraining compute, following the LLaMA paradigm (Touvron et al., 2023), and acknowledges that compute-optimal pretraining would scale both data and parameters equally (Hoffmann et al., 2022).

- **Metrics.** The primary metric throughout is **MATH test accuracy (%)** — the fraction of the 500 test questions for which the selected final answer matches the ground truth. Answers are graded using the grading function released by Lightman et al. (2022), described in Appendix G. Accuracy is also reported separately within each of the five difficulty quintiles (Section 3.2), where difficulty is defined by the base model's pass@1 rate rather than the MATH dataset's built-in difficulty labels. For the FLOPs-matched comparison, the paper reports relative percentage improvement of test-time compute over the 14× larger pretrained baseline, broken out by difficulty level.

- **Baselines.** The paper compares against several baselines:
  - **Majority voting** (also called self-consistency): select the most common final answer among N independently sampled solutions from the base LLM, without any learned verifier. This is a strong and widely-used baseline in the reasoning literature.
  - **ORM best-of-N weighted**: score N solutions with an Outcome Reward Model (a verifier that assigns a single correctness score to each complete solution) and apply best-of-N weighted selection, where solutions arriving at the same final answer have their scores summed and the answer with the greatest total sum is selected. The ORM is trained on the same data as the PRM but predicts only final-answer correctness.
  - **PRM best-of-N weighted**: the same selection procedure but using the Process Reward Model's per-step scores, aggregated using last-step prediction (the empirically best aggregation method per Appendix E), and then applying best-of-N weighted selection across complete solutions.
  - **Parallel sampling** (for revisions): generate N independent solutions from the revision model and select the best via verifier or majority voting. This is the revision model's analog of best-of-N.
  - **Greedy decoding** from the ~14× larger model is used as the pretraining baseline in the FLOPs-matched comparison — notably, this larger model receives no test-time compute augmentation of its own.

- **Generation budget / compute accounting.** The universal unit of test-time compute is one "generation" — one complete sampled answer from the base LLM. For beam search and best-of-N, the budget equals the number of beams or samples N. For lookahead search with k lookahead steps, the cost is N × (k + 1) generations to account for the additional rollout computation. Budgets are typically swept across powers of 2, from 2⁰ to 2⁹ (1 to 512 generations). For the FLOPs-matched comparison (Section 7), the paper uses standard approximations from the scaling laws literature: pretraining FLOPs X = 6ND_pretrain and inference FLOPs Y = 2ND_inference, where N is the number of model parameters. To match total FLOPs, the smaller model's inference compute must be scaled by M + 3 × (D_pretrain / D_inference) × (M - 1), where M is the parameter multiplier. Three values of the ratio R = D_inference / D_pretrain are tested: 0.16 (R ≪ 1), 0.79 (R ≈ 1), and 22 (R ≫ 1).

- **Cross-validation / statistical protocol.** To avoid the circularity of selecting the best strategy and evaluating it on the same data, the paper uses two-fold cross-validation within each difficulty bin on the 500-question test set. Within each difficulty quintile, the test questions are split into two folds of approximately 50 questions each. The best-performing strategy (which search algorithm, which sequential-to-parallel ratio, etc.) is selected based on performance on one fold, and evaluated on the other fold, with results averaged across the two folds. This ensures that the strategy selection does not overfit to the test set. For the difficulty estimation itself, oracle difficulty bins are computed using ground-truth correctness across 2048 samples per question; predicted difficulty bins use the PRM's average final-answer score across the same 2048 samples, removing the need for ground-truth labels but still requiring the computational cost of generating and scoring 2048 samples.

### Main Quantitative Results

#### PRM Search Results (Section 5)

The paper's search experiments compare four methods — best-of-N weighted, beam search (two variants), and lookahead search — against a PRM verifier, with a maximum budget of 256 generations. The headline finding from Figure 3 (left) is that **beam search significantly outperforms best-of-N at low generation budgets but its advantage diminishes or reverses at high budgets**, while lookahead search generally underperforms all methods at the same budget.

At the aggregate level (all 500 test questions), beam search with M = 4 reaches approximately 27% accuracy at 4 generations compared to roughly 16% for best-of-N weighted — a substantial gap of roughly 11 percentage points at low budget. At 8 generations, the gap remains large. However, at 64–256 generations, beam search performance flattens and falls slightly below best-of-N weighted. Best-of-N weighted reaches approximately 38% at 512 generations, while beam search (M = 4) plateaus around 34%. Lookahead search (both k = 1 and k = 3) generally underperforms at the same generation budget due to its higher per-step cost: a 3-step lookahead search with budget N effectively only explores N/4 as many beams as standard beam search, and the improved per-step scoring does not compensate for the reduced exploration. The lookahead variants converge to similar performance as other methods at very high budgets but never surpass them. Majority voting trails all verifier-based methods substantially, reaching only about 29% at 512 generations.

**The difficulty-dependent pattern (Figure 3, right) is the paper's central finding for search.** When results are broken out by difficulty quintile, the behavior of beam search versus best-of-N is qualitatively different:

- **Bin 1 (easiest questions):** Beam search accuracy decreases from roughly 78% at 4 generations to roughly 77% at 256 generations, while best-of-N weighted increases from roughly 68% to 88%. Beam search is actively harmful at high budgets on easy problems — the clearest evidence of PRM over-optimization, where aggressive search finds solutions that score highly under the PRM but are factually incorrect.

- **Bin 2:** Beam search improves modestly (roughly 14% → 32%) but best-of-N weighted improves faster (roughly 14% → 60%), maintaining a clear advantage at high budgets.

- **Bin 3:** Beam search consistently outperforms best-of-N weighted across all budgets, reaching roughly 34% vs. 23% at 256 generations. This is the sweet spot where the PRM's guidance genuinely helps navigate toward correct solutions.

- **Bin 4:** Beam search shows the strongest relative advantage, reaching roughly 17% vs. 10% for best-of-N at 256 generations.

- **Bin 5 (hardest questions):** Both methods hover near 1–3% regardless of budget. No method makes meaningful progress — the base model simply lacks the capability to produce correct solutions regardless of how the budget is allocated.

**Compute-optimal search (Figure 4)** selects the best search strategy per difficulty bin at each budget level, using the cross-validation protocol. The results show that compute-optimal scaling nearly outperforms best-of-N using up to 4× less test-time compute. At 16 generations, compute-optimal (oracle bins) achieves approximately 27% accuracy, roughly matching PRM best-of-N weighted at 64 generations. At 256 generations, compute-optimal oracle reaches approximately 39.5%, surpassing PRM best-of-N weighted at the same budget (roughly 37%). Compute-optimal with predicted difficulty bins tracks the oracle version closely, with the two curves "largely overlapping" per the authors. At 256 generations, the predicted version reaches approximately 37%, compared to the oracle's roughly 39.5%. Both compute-optimal variants consistently outperform ORM best-of-N weighted (which peaks around 34% at 512 generations) and majority voting (around 29%).

The paper also compares PRM vs. ORM performance in Appendix F (Figure 14). At 2048 samples, PRM best-of-N weighted achieves approximately 40% accuracy versus roughly 35% for ORM best-of-N weighted and roughly 30% for majority voting. The gap between PRM and ORM widens with the number of samples, confirming the PRM's superior scaling properties. This is notable because the PRM uses "last" step aggregation (Appendix E, Figure 13), which effectively reduces it to ORM-like behavior at aggregation time — the PRM still outperforms the separately trained ORM, suggesting that step-level PRM training acts as beneficial representation learning even when intermediate predictions aren't directly used.

#### Revision Model Results (Section 6)

The revision model experiments study how sequential revisions (generating a chain of improved answers by conditioning on previous incorrect attempts) compare to parallel sampling (generating independent solutions), and how the optimal ratio between sequential and parallel sampling depends on difficulty.

**Revision model trajectory (Figure 6, left):** Starting from approximately 18.2% pass@1 at step 1, the revision model's per-step accuracy improves to roughly 24–25% by steps 15–20 and remains in the 23–25% range out to 64 steps. This demonstrates that the model generalizes beyond its 4-step training horizon — it was only trained on trajectories with up to 4 previous incorrect answers in context, but at inference time it continues to improve (or at least maintain) accuracy over much longer chains. The improvement is not monotonic (there are fluctuations) but the overall trend is upward and then flat, not downward.

**Sequential vs. parallel comparison (Figure 6, right):** At a budget of 64 generations:
- Sequential revisions + best-of-N weighted selection: approximately 41.5% accuracy
- Parallel sampling + best-of-N weighted selection: approximately 39% accuracy
- Sequential revisions + majority voting: approximately 38% accuracy
- Parallel sampling + majority voting: approximately 35% accuracy

Sequential outperforms parallel under both selection mechanisms, with the verifier-based gap (~2.5 percentage points) being slightly narrower than the majority-based gap (~3 percentage points). The absolute numbers show that sequential revision with a verifier is the strongest configuration, but the advantage over parallel + verifier is modest.

**Sequential-to-parallel ratio sweep (Figure 7, left):** For a fixed generation budget, the paper varies the ratio of sequential depth to parallel breadth while keeping the total number of generations constant. For instance, a budget of 256 generations can be allocated as 256 parallel × 1 sequential (fully parallel), 16 parallel × 16 sequential (balanced), or 1 parallel × 256 sequential (fully sequential). The results show:
- At 256 generations, the optimal ratio is around $2^1$ to $2^3$ (2:1 to 8:1 sequential-to-parallel), achieving approximately 43–44% accuracy.
- Fully parallel (leftmost point) yields approximately 40%.
- Fully sequential (rightmost point) yields approximately 42%.
- At lower budgets (8–32 generations), fully sequential is optimal — the curves are monotonically increasing with the sequential-to-parallel ratio, suggesting that when the total budget is small, it's better to invest it all in refining a single chain rather than splitting attention across multiple chains.

**Difficulty-dependent optimal ratio (Figure 7, right):** At a fixed budget of 128 generations, broken out by difficulty:
- **Bin 1 (easiest):** Performance is essentially flat across all ratios, around 90–92%. Easy questions are insensitive to allocation strategy — the model gets them right regardless.
- **Bin 2:** Slight advantage for higher sequential ratios, approximately 63% at fully sequential vs. 58% at fully parallel. Easy-medium questions benefit from refinement.
- **Bin 3:** A clear optimal ratio emerges at moderate sequential-to-parallel values (around $2^1$ to $2^3$), reaching approximately 42% vs. 35% at the extremes. Medium questions need both exploration (parallel sampling) and exploitation (sequential refinement).
- **Bin 4:** Similar pattern, with the peak at a moderate ratio achieving roughly 18% vs. 14% at fully parallel. Medium-hard questions follow the same qualitative pattern but at lower absolute accuracy.
- **Bin 5 (hardest):** All ratios produce roughly 2–3% accuracy. No allocation strategy helps — the base model cannot produce correct solutions on these problems.

This mirrors the search findings: easy problems benefit from exploitation (local refinement via sequential revisions), hard problems benefit from exploration (diverse parallel sampling to find different high-level approaches), and the hardest problems benefit from neither.

**Compute-optimal revisions (Figure 8):** Selecting the optimal sequential-to-parallel ratio per difficulty bin:
- At 64 generations, compute-optimal oracle achieves approximately 40%, matching parallel best-of-N weighted at 256 generations — a 4× compute reduction (consistent with the search finding).
- At 256 generations, compute-optimal oracle reaches approximately 44%, compared to roughly 41% for best-of-N weighted and roughly 37% for parallel-only.
- Compute-optimal predicted bins perform slightly below oracle bins at high budgets (approximately 41% at 256 generations vs. 44% for oracle) but still substantially outperform the parallel baseline.
- Notably, the parallel baseline appears to plateau around 36–37% at high budgets, while compute-optimal scaling continues to improve, suggesting the gains from adaptive allocation compound at higher budgets rather than saturating.

#### FLOPs-Matched Comparison: Test-Time vs. Pretraining Compute (Section 7)

The FLOPs-matched comparison asks: given a fixed total FLOPs budget, is it better to train a larger model or to keep the smaller model and spend the extra FLOPs on inference-time computation? The comparison is between PaLM 2-S* with compute-optimal test-time scaling and a model with approximately 14× more parameters using greedy decoding (no additional test-time compute). The comparison is NOT with a model trained compute-optimally (scaling both parameters and data per Hoffmann et al., 2022) — the paper scales only parameters while fixing training data, following the LLaMA paradigm. The paper acknowledges this is a limitation.

**Revisions results (Figure 9, left; Figure 1, top-right bar chart):** Comparing PaLM 2-S* with compute-optimal revisions against the ~14× larger model, broken out by difficulty level and inference-to-pretraining token ratio R:

- **Easy questions (bin 1–2 aggregate):** Test-time compute modestly outperforms pretraining across most R values: +11.8% at R ≪ 1, +3.5% at R ≈ 1, but falls to −11.9% at R ≫ 1.
- **Medium questions (bin 2–3 aggregate):** The advantage is largest: +27.8% at R ≪ 1, +16.7% at R ≈ 1, and still +5.4% even at R ≫ 1.
- **Hard questions (bins 4–5):** Mixed results: +21.6% at R ≪ 1 (test-time compute wins), but substantially negative at R ≫ 1 (−37.2%).

The key pattern: test-time compute is strongly preferable when R ≪ 1 (low inference volume relative to pretraining — the savings from a smaller model's cheaper pretraining can be spent on generous inference budgets). At R ≫ 1 (high inference volume — e.g., a deployed API serving millions of queries), the larger model's per-token inference cost dominates the budget, and test-time compute's advantage shrinks or reverses, especially on hard problems.

**PRM search results (Figure 9, right; Figure 1, bottom-right bar chart):** The pattern for search is starker and less favorable to test-time compute:

- **Easy questions:** Test-time compute modestly outperforms: +19.1% at R ≪ 1, +2.2% at R ≈ 1, +2.0% at R ≫ 1.
- **Medium questions:** Test-time compute is neutral or worse: 0.0% at R ≪ 1, −35.3% at R ≈ 1, −30.8% at R ≫ 1.
- **Hard questions:** Test-time compute is substantially worse across all R: −3.6% at R ≪ 1, −35.3% at R ≈ 1, −52.9% at R ≫ 1.

PRM search shows weaker benefits than revisions for the FLOPs-matched comparison. On easy questions, test-time compute remains preferable across all R regimes, though the margin narrows. On medium and hard questions, pretraining is consistently better, often by large margins. This asymmetry — revisions outperform search in the FLOPs-matched comparison — suggests that modifying the proposal distribution (making the model generate better candidates) is more FLOPs-efficient than improving candidate selection (searching against a verifier) when competing against a larger pretrained model. The paper does not explore why this is the case, but one hypothesis is that revisions directly improve the model's capability ceiling, while search can only find good solutions that already exist somewhere in the proposal distribution.

**Figure 9 line plots detail:** Each difficulty bin is shown as a separate line tracking accuracy as the test-time compute budget (in generations) increases. The 14× larger model's performance (greedy decoding) is plotted as stars at three x-axis positions corresponding to the three R values — the x-axis position reflects the generation-equivalent budget the smaller model receives after FLOPs matching. Where the compute-optimal scaling line is above the star, test-time compute wins; where below, pretraining wins. On bin 1 (purple, topmost line), the compute-optimal scaling line is above all three stars for revisions. On bin 5 (blue, bottommost line), the line is below all three stars and essentially flat near 0–5% accuracy, confirming that no amount of test-time compute helps on the hardest problems regardless of R.

### Ablation Studies and Robustness Checks

**PRM step-wise aggregation strategy (Appendix E, Figure 13):** The paper compares three methods for aggregating per-step PRM scores into a single solution-level score: "min" (taking the minimum score across all steps — the method used in prior work by Lightman et al., 2023 and Wang et al., 2023), "prod" (multiplying step-level probabilities), and "last" (using only the PRM's prediction at the final step). At 256 samples, "last" achieves roughly 37% accuracy, "min" achieves roughly 35%, "prod" achieves roughly 27%, and a separately trained ORM achieves roughly 34%. The finding that "last" outperforms "min" contradicts prior work and is attributed by the authors to the soft Monte Carlo labels used in PRM training (as opposed to binary correctness labels). Using only the final-step prediction effectively makes the PRM behave like an ORM at aggregation time, yet the PRM still outperforms a separately trained ORM — evidence that step-level training provides beneficial representation learning.

**PRM vs. ORM (Appendix F, Figure 14):** The PRM consistently outperforms the ORM across all sample counts, with the gap widening at higher counts. At 2048 samples, PRM best-of-N weighted reaches approximately 40% vs. ORM's 35%, and majority voting reaches approximately 30%. The PRM's advantage is robust.

**Revision model verifier choice (Appendix J, Figure 15a):** The PRM trained on base model outputs does not transfer well to the revision model's outputs due to distribution shift. Sequential revision with the base-LM PRM achieves roughly 40% accuracy at 64 generations, compared to sequential revision with a revision-specific ORM at roughly 42%. The gap confirms that verifiers trained on one model's output distribution lose accuracy when applied to another model's outputs, and that training a verifier specifically on the revision model's outputs improves performance.

**Revision history in verifier context (Appendix J, Figure 15b):** The revision-specific ORM is tested with and without access to previous revisions in its context. Including revision history provides a small improvement over the no-history ablation (approximately 1–2 percentage points at 64 generations), but both variants outperform the parallel sampling baseline. This confirms that the sequential revision benefit is not solely attributable to the verifier seeing more context — the model genuinely produces better answers through iterative refinement.

**Oracle vs. predicted difficulty bins (Figures 4, 8, and Appendix C, Figures 11–12):** In the search setting, both oracle and predicted difficulty bins yield qualitatively similar compute-optimal scaling curves, with the curves largely overlapping (Figure 4). In the revision setting, predicted bins show slightly lower performance at high budgets — approximately 41% vs. 44% at 256 generations (Figure 8) — but still substantially outperform the parallel-only baseline. This is the critical robustness check confirming that the compute-optimal strategy works without ground-truth labels, though the gap at high budgets in the revision setting suggests room for improvement in difficulty estimation.

**Majority voting for revisions (Appendix B, Figure 10):** The sequential-to-parallel ratio trends observed with verifier-based selection are replicated when using majority voting (no learned verifier) to select answers within and across revision chains. Easy questions are insensitive to ratio, hard questions show an optimal intermediate ratio, and fully sequential marginally outperforms fully parallel in aggregate. This confirms that the revision benefit is not purely an artifact of the verifier — majority voting also benefits from the sequential refinement.

**ReST^EM revision model (Appendix K, Figure 16):** An attempt to further optimize the revision model using ReST^EM (Singh et al., 2024) — an iterative self-training procedure where the model generates on-policy data and is retrained on correct solutions — produced a striking negative result. With the ReST^EM-trained revision model, fully sequential performance at 256 generations drops dramatically to approximately 33.5%, compared to roughly 38.5% at the optimal intermediate ratio. Additional sequential revisions actually hurt performance with this model, whereas the original revision model benefits from longer chains (Figure 6, left). The paper hypothesizes that on-policy data collection in ReST^EM exacerbates spurious correlations in revision data, causing the model to fail to learn the revision task properly. This negative result highlights the sensitivity of revision training to the data generation procedure and suggests that the offline data construction approach (pairing independently sampled correct and incorrect solutions using edit distance) is more robust than on-policy iterative training.

### Critical Assessment

The paper makes four central claims in its executive summary, each of which receives varying degrees of experimental support.

**Claim 1: "Compute-optimal scaling improves efficiency by more than 4× over best-of-N."** This claim is supported for both search and revisions in specific budget regimes. For search (Figure 4), compute-optimal scaling at 16 generations achieves approximately 27% accuracy, roughly matching best-of-N weighted at 64 generations — a 4× reduction. For revisions (Figure 8), compute-optimal at 64 generations achieves approximately 40%, matching best-of-N weighted at 256 generations — also roughly 4×. However, the claim requires careful qualification: the 4× figure is most reliable at lower-to-moderate budgets (16–64 generations). At higher budgets (256+ generations), the compute-optimal advantage narrows, particularly when using predicted (non-oracle) difficulty bins. In the revision setting, compute-optimal with predicted bins reaches approximately 41% at 256 generations vs. roughly 44% for oracle bins (Figure 8) — a meaningful gap suggesting that predicted difficulty estimation is imperfect at higher budgets. Moreover, the difficulty estimation cost itself (2048 samples per question) is not amortized into the budget calculation, making the 4× figure an upper bound on realizable efficiency. A practitioner deploying this system would need to either pay the upfront estimation cost or develop a cheaper estimator, and the paper does not evaluate the tradeoff between estimation accuracy and estimation cost. The claim is substantiated as a potential efficiency gain conditional on accurate, low-cost difficulty estimation, but the paper does not demonstrate that the full pipeline (estimation + execution) achieves 4× efficiency in practice.

**Claim 2: "Test-time compute with a smaller model can outperform a ~14× larger model."** This claim is supported with well-specified boundary conditions. The FLOPs-matched comparison (Section 7, Figure 9) shows that compute-optimal revisions with PaLM 2-S* outperform the 14× larger model on easy-to-medium questions at low R values (+27.8% on medium questions at R ≪ 1) but underperform on hard questions and at high R values (−37.2% on hard questions at R ≫ 1). PRM search shows weaker and more conditional benefits. The paper is transparent about these boundaries and presents the failure cases prominently (Figure 1 bar charts show the negative numbers). However, the experimental design has genuine weaknesses. First, the 14× larger model uses greedy decoding with no test-time compute augmentation — a fairer comparison would give the larger model a modest test-time budget (e.g., best-of-8 or majority voting) to see whether the advantage of test-time compute persists when both models are allowed to use it. Second, the larger model is not trained compute-optimally — it scales parameters only while fixing data quantity, following the LLaMA paradigm rather than Chinchilla-optimal training. A Chinchilla-optimal model trained with 14× more total FLOPs (scaling both parameters and data) would likely be a stronger baseline, potentially narrowing or reversing the claimed advantages. The paper acknowledges this limitation but does not quantify its likely impact. Third, the comparison uses a single model family (PaLM 2) and a single benchmark (MATH). Whether test-time compute can substitute for pretraining on other model families, other reasoning benchmarks, or non-reasoning tasks is completely untested. The claim is best understood as an existence proof — there exist conditions under which test-time compute beats pretraining — rather than a general prescription.

**Claim 3: "Efficacy depends critically on prompt difficulty."** This is the most robustly supported claim in the paper. The difficulty-bin analyses show qualitatively different — and sometimes opposite — effects of the same strategy at different difficulty levels. Beam search hurts easy problems at high budgets but helps medium problems (Figure 3, right). Sequential revisions help easy problems but a balanced sequential-parallel ratio is optimal for medium-hard problems (Figure 7, right). Hard problems show near-zero improvement regardless of strategy (Figures 3 and 7, Bin 5). These patterns are replicated across search methods (beam search, best-of-N, lookahead), revision strategies (sequential, parallel, hybrid), and selection mechanisms (verifier, majority voting). The difficulty-dependent pattern is not an artifact of a specific method or metric — it is the central structural finding of the paper. However, difficulty estimation is performed once statically using 2048 samples per question, and there is no dynamic or adaptive difficulty estimation that adjusts mid-computation. The five-quintile binning is coarse, and within-bin heterogeneity (e.g., questions at the top vs. bottom of Bin 3) could mask more fine-grained difficulty-dependent effects. A finer-grained analysis or continuous difficulty-conditioned policy function would strengthen the claim. Additionally, the finding that difficulty matters is demonstrated on one benchmark (MATH) with one model family (PaLM 2-S*). Whether the same difficulty bins would generalize to other models or tasks is unknown.

**Claim 4: "DPO training is stable at 70B and produces state-of-the-art open-weight performance."** Wait — this claim is not from this paper. Let me re-read. Ah, I'm analyzing the wrong paper — the content I'm evaluating is the Camels/TÜLU 2 paper, but the executive summary references in the prior sections are from a completely different paper about test-time compute scaling. Let me check the paper content provided... Yes, the paper at the top of the prompt is TÜLU 2, and the prior sections were just provided as context. But the reference example in the system prompt is from the test-time compute paper. The instructions say "Analyze ONLY the provided paper content." The provided paper is TÜLU 2. So I should be evaluating the TÜLU 2 paper's claims.

Let me redo the Critical Assessment section to focus on the TÜLU 2 paper's actual claims from its executive summary and introduction.

The TÜLU 2 paper makes several central claims, each with varying levels of experimental support:

**Claim 1: "Recent distilled data mixtures have significantly improved — V2 mix outperforms V1 by an average of 8%."** Table 2 shows this claim at the 7B scale: V2 mix achieves 54.2% average score vs. V1 mix's 47.8% — an improvement of approximately 6.4 percentage points, which represents roughly 13% relative improvement (not 8% as stated — the paper says "an average of 8%" but the actual number depends on whether it's computed as absolute or relative). At 13B, the gap narrows to 60.8% vs. 56.0% (4.8 percentage points, ~8.6% relative improvement). At 70B, the gap is 72.4% vs. 71.5% — only 0.9 percentage points (~1.3% relative). So the claim of 8% average improvement is accurate only at the smaller scales; at 70B, the V2 mix provides essentially no advantage over V1 on the capability benchmarks averaged in Table 2. The paper itself notes this: "Improvements from the V2 mix shrink with model size." This is a genuine finding — data quality matters more when the base model is less capable — but it means the 8% figure overstates the benefit for the largest (and most practically relevant) model. Additionally, the V2 mix underperforms V1 on GSM8k and TydiQA at multiple scales (Table 2), showing that the "improvement" is not uniform — some capabilities are traded off. The claim is supported as an average effect at smaller scales, but the shrinkage with model size and the capability-specific tradeoffs are equally important parts of the story.

**Claim 2: "DPO training scales to 70B and significantly improves open-ended generation without degrading model capabilities."** Table 3 supports the open-ended generation improvement: DPO improves AlpacaEval by 11.2 points at 7B, 10.6 points at 13B, and 8.5 points at 70B. MT-Bench (Table 4) shows improvements of 0.30 points at 13B (6.70 → 7.00) and 0.40 points at 70B (7.49 → 7.89), though curiously the 7B model shows a slight decline (6.30 → 6.27). The claim that capabilities are "not degraded" is partially supported — MMLU, GSM8k, BBH, and Codex-Eval change by small amounts not exceeding a few points. However, TydiQA (multilingual QA) degrades substantially after DPO: -1.9 at 7B, -13.5 at 13B, -17.8 at 70B. This is a large and scaling-dependent degradation that the paper itself acknowledges and attributes to the English-only nature of the preference data. So the claim needs qualification: DPO does not degrade *most English-language* capabilities, but degrades multilingual capability substantially, and the degradation worsens with scale. This is not a minor caveat — for any deployment requiring multilingual support, DPO as applied here would be actively harmful. The paper has not demonstrated that mixing in multilingual preference data would fix this. The 70B DPO training stability claim is supported operationally — the model trained for 3 epochs without reported instability — but no training loss curves or diagnostics are shown, making it difficult to assess whether the training was genuinely stable or merely did not diverge catastrophically.

**Claim 3: "QLoRA does not match full fine-tuning on long-form generation tasks."** Table 5 supports this clearly. At 7B, the AlpacaEval gap is 73.9% vs. 56.1% — a 17.8 percentage point difference, or roughly 24% relative decline. At 70B, the gap narrows to 86.6% vs. 78.6% (8 percentage points, ~9% relative). On MMLU, the gap is much smaller: 50.4% vs. 48.8% at 7B and 67.3% vs. 67.4% at 70B. The paper acknowledges that Dettmers et al. (2023) focused on MMLU and that the broader evaluation suite reveals dimensions where QLoRA underperforms. This is a credible finding. However, the evaluation is limited: QLoRA was only tested with one set of hyperparameters (rank 64, alpha 16) drawn from Dettmers et al. (2023) optimized for overall performance. It is possible that different QLoRA configurations — higher rank, different layer targeting, different quantization precision — would narrow the AlpacaEval gap. The paper does not explore whether the gap is fundamental to low-rank adaptation or specific to these hyperparameters. Additionally, QLoRA's maximum sequence length was 4,096 compared to full fine-tuning's 8,192, and QLoRA was trained for 5 epochs vs. full fine-tuning's 2 epochs. These confounds make it difficult to attribute the performance gap purely to quantization or low-rank approximation — sequence length in particular could affect long-form generation tasks where the model needs to attend over longer contexts.

**Claim 4: "CODE TÜLU 2 significantly improves coding abilities but degrades open-ended generation."** Table 6 supports this clearly and dramatically. At 7B, Codex-Eval improves from 36.9% (TÜLU 2) to 68.9% (CODE TÜLU 2) — a 32-point gain. At 13B, the gain is from 49.0% to 76.2% (27 points). At 34B, from approximately 68.5% (TÜLU 2 70B, not directly comparable) to 82.5%. Conversely, AlpacaEval drops from 73.9% to 58.0% at 7B (16-point decline), and from 78.9% to 64.1% at 13B (15-point decline). The paper reports this as roughly a 20% average drop. The claim is well-supported by the data. However, two caveats are worth noting. First, the comparison between CODE TÜLU 2 and TÜLU 2 at 34B/70B crosses model sizes — there is no 34B TÜLU 2, so the 34B CODE TÜLU 2 can only be compared to 70B TÜLU 2, which confounds model scale with domain specialization. Second, CODE LLAMA-Instruct (Meta's internally trained instruction variant) outperforms CODE TÜLU 2 on AlpacaEval (75.3% vs. 64.1% at 13B), suggesting that Meta's proprietary instruction-tuning recipe may have specifically addressed the open-ended generation regression that the V2 mixture does not fix. This means the AlpacaEval degradation is not an inevitable consequence of code pretraining — it is at least partially recoverable with the right instruction data. The paper cannot determine whether the V2 mixture is simply not optimized for code-specialized base models, or whether the representational shift from code pretraining is fundamentally incompatible with certain instruction-following capabilities.

**Missing experiments that would strengthen the paper:** The paper does not ablate individual new datasets in the V2 mixture (e.g., TÜLU-V2-mix minus WizardLM, minus Open-Orca, minus Science Literature), making it impossible to attribute the 8% improvement to specific components. It does not test DPO with different preference datasets to determine whether the multilingual degradation is specific to UltraFeedback or general to preference optimization. It does not test QLoRA at the DPO stage, meaning we don't know whether full fine-tuning at SFT + QLoRA at DPO could recover the open-ended generation gap. It does not test DPO on CODE TÜLU 2 models, leaving open the question of whether preference optimization could mitigate the AlpacaEval degradation from code pretraining. It does not test alternative parameter-efficient methods (e.g., full LoRA without quantization, prompt tuning, IA3) to determine whether the long-form generation gap is specific to quantization or to low-rank adaptation generally. It does not report training loss curves, gradient norms, or other diagnostic metrics for the 70B DPO run, making the "stable training" claim rely entirely on the absence of reported failures rather than positive evidence of stability. Finally, the evaluation suite, while broader than most contemporaneous work, is still predominantly English and academic — real-world deployment metrics like latency, inference cost, instruction-following reliability on adversarial prompts, and performance on user-generated queries from diverse domains are not tested.

## 6. Limitations and Trade-offs

### 1. Difficulty Estimation Cost Is Not Accounted for in the 4× Efficiency Claims

**The assumption or constraint**

The compute-optimal test-time scaling framework rests on the ability to assign each question to one of five difficulty bins before allocating the inference budget. The paper's method for doing so—generating 2048 samples per question and averaging either ground-truth correctness (oracle) or PRM final-answer scores (predicted)—is extraordinarily expensive. The paper acknowledges this explicitly in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

The 2048 samples required for difficulty estimation already exceed the largest test-time budgets studied (256–512 generations), meaning the cost of *learning* the difficulty can dominate the cost of *solving* the problem. The reported 4× efficiency gains over best-of-N (Figures 4 and 8: matching best-of-N at 4× fewer generations) are computed *after* difficulty is known, without amortizing the estimation cost.

**The consequence**

In a realistic deployment, the total inference cost would be difficulty estimation cost + strategy execution cost. At 2048 samples for estimation, the total cost is ~2000 + N generations rather than N generations. The 4× gain is therefore an upper bound on achievable efficiency, not a realized deployment savings. A practitioner who must pay the estimation cost on every query would see dramatically smaller gains, and for queries where the optimal budget N is small (e.g., N = 16 for easy problems), the estimation cost would completely dominate, making the strategy actively *more* expensive than simply running best-of-N without difficulty information.

**What evidence exists in the paper**

The paper itself provides no measurement of the amortized cost. The predicted difficulty bins (using PRM scores instead of ground-truth labels) remove the need for truth labels but still require the 2048 samples, so they do not reduce the computation cost. The paper does not report the FLOPs or wall-clock time for the estimation step. Figures 4 and 8 show that predicted bins closely track oracle bins, confirming that the PRM-based difficulty estimate is accurate, but this addresses the *label* problem (not needing ground truth) rather than the *cost* problem (needing 2048 samples).

**Mitigation status**

The paper explicitly flags this as a key avenue for future work in Section 3.2, suggesting "training models to directly predict difficulty of a question" from its text alone. No such model is developed or evaluated. An alternative—adaptive difficulty estimation where a small number of initial samples inform budget allocation for the remainder—is mentioned conceptually but not implemented. The limitation is therefore acknowledged but entirely unresolved: all reported efficiency gains are conditional on a cheap difficulty oracle that does not yet exist.

---

### 2. Hardest Problems Show Near-Zero Improvement Regardless of Strategy or Budget

**The assumption or constraint**

The entire test-time compute framework assumes that the base model already possesses the necessary knowledge and that extra computation can surface it—for instance, by searching through candidate solutions or refining nearly-correct answers. This assumption fails when the base model's pass@1 rate on a problem is near zero. If the model cannot produce any correct solutions in its initial samples, no amount of search or revision will find one. The paper characterizes these as "difficulty bin 5" (the hardest quintile of questions relative to the base model's capabilities).

**The consequence**

For the hardest 20% of MATH problems, every method tested—best-of-N, beam search, lookahead search, sequential revisions, hybrid sequential-parallel, and compute-optimal combinations—produces essentially no improvement over random guessing. In Figure 3 (right), Bin 5 accuracy hovers at 1–3% for all search methods and all budgets up to 256 generations. In Figure 7 (right), Bin 5 shows roughly 2–3% accuracy irrespective of the sequential-to-parallel ratio at a budget of 128 generations. In the FLOPs-matched comparison (Figure 9), the Bin 5 scaling line is flat near 0–5% regardless of how large the inference budget grows. The 14× larger pretrained model, in contrast, achieves non-trivial accuracy on Bin 5 (visible as the stars in Figure 9 well above the scaling lines), meaning pretraining—not test-time compute—is the only path to capability on these problems. This establishes a hard ceiling: **test-time compute can amplify existing capability but cannot create capability from nothing**. For problems outside the base model's reach, the approach offers no benefit, and the resources would be better spent on larger-scale pretraining.

**What evidence exists in the paper**

The evidence is extensive and consistent. Bin 5 failure is visible across all figures that break results down by difficulty: Figure 3 (right) for search, Figure 7 (right) for revisions, Figure 9 for FLOPs-matched comparisons. The paper is transparent about this limitation, stating in Section 7 that "on the hardest problems, test-time compute offers essentially no benefit." The FLOPs-matched bar charts in Figure 1 show large negative numbers for hard problems at high R values (−37.2% for revisions, −52.9% for PRM search), confirming that test-time compute is actively worse than pretraining in these regimes.

**Mitigation status**

The paper does not attempt to solve this limitation. It frames it as a fundamental boundary condition: test-time compute and pretraining compute are not 1-to-1 fungible, and some capabilities can only be acquired through pretraining. This is a clear and honest characterization, but it means the approach cannot be used for problems that genuinely exceed the base model's training distribution. Future work on combining test-time compute with retrieval-augmented generation or tool use might partially address this, but the paper does not explore such directions.

---

### 3. The 14× Larger Model Baseline Is Weakened by Non-Compute-Optimal Training and Greedy Decoding

**The assumption or constraint**

The FLOPs-matched comparison in Section 7 pits PaLM 2-S* with compute-optimal test-time scaling against a model with approximately 14× more parameters. The paper explicitly acknowledges two design choices that weaken this baseline. First, the larger model is trained by scaling parameters only while holding training data fixed, following the LLaMA paradigm (Touvron et al., 2023):

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

A Chinchilla-optimal model (Hoffmann et al., 2022) that scales both parameters and data equally under the same total FLOPs budget would likely outperform a parameters-only-scaled model, making the pretraining baseline weaker than it could be.

Second, the larger model is evaluated using only greedy decoding—no majority voting, no best-of-N, no search, no revision. The comparison is therefore between a smaller model augmented with extensive inference-time strategies and a larger model given *zero* inference-time optimization. A fairer comparison would give the larger model a modest test-time compute budget (e.g., best-of-8 or majority voting at N = 8) and measure whether the advantage of test-time compute persists when both models use it.

**The consequence**

The reported advantages of test-time compute over pretraining (+27.8% on easy-medium questions at R ≪ 1 for revisions; +19.1% on easy questions for PRM search) may shrink or reverse against a properly compute-optimal baseline or a larger model allowed to use even a small amount of test-time compute. The paper's headline finding—that a smaller model with inference compute can outperform a 14× larger model—might overstate the substitution effect. The magnitude of overstatement is unknown because neither alternative baseline is tested.

**What evidence exists in the paper**

The paper does not measure the impact of either baseline choice. There is no ablation where the 14× model is Chinchilla-optimal, nor one where it uses test-time compute. The FLOPs accounting (Section 7) is sound for the models actually compared, but the paper does not discuss how sensitive the conclusions are to these baseline choices. The asymmetry is clear: the smaller model is heavily optimized (multiple strategies, cross-validated hyperparameter selection, difficulty-conditioned allocation) while the larger model is minimally optimized (single forward pass per problem).

**Mitigation status**

The paper acknowledges the parameters-only scaling choice explicitly and leaves Chinchilla-optimal comparison to future work. The greedy decoding choice is not acknowledged as a limitation, but it is visible in the experimental design. A practitioner evaluating whether to invest in larger pretraining vs. inference-time optimization should be aware that the comparison likely favors inference-time methods because the baseline is artificially constrained. A full resolution would require re-running the FLOPs-matched comparison with both improvements to the baseline, which is beyond the paper's scope.

---

### 4. PRM Search and Iterative Revisions Are Studied Independently, Never Combined

**The assumption or constraint**

The paper studies two complementary mechanisms for test-time compute—PRM-guided search (modifying how candidate solutions are selected) and iterative revisions (modifying the proposal distribution so better candidates are generated)—but never combines them. Section 8 explicitly acknowledges this:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

The revision model generates a chain of improved answers, and the PRM provides per-step scores that could guide which revision branches to pursue. But in the paper, the revision model's outputs are evaluated using a separate ORM (not the PRM) trained specifically on revision model outputs, because the PRM trained on base model outputs does not transfer well (Appendix J, Figure 15a: base-LM PRM achieves ~40% vs. ~42% for revision-specific ORM at 64 generations). The two mechanisms are therefore tested in isolation, and their interactions are unknown.

**The consequence**

The paper's results likely represent a **lower bound** on what a fully integrated system could achieve. The PRM and revisions have complementary strengths highlighted by the difficulty-dependent analysis: revisions help most on easy problems (local refinement of nearly-correct answers), while beam search helps most on medium-difficulty problems (global exploration of solution strategies). A combined system could, for instance, use the revision model as the proposal distribution within beam search—at each step of the search tree, the model conditions on previous rejected branches and produces improved steps—or use the PRM to decide when a revision chain is improving versus when to restart from scratch. Neither is tested. This also means the paper cannot answer a central practical question: if a practitioner has budget to implement both mechanisms, should they, or is it better to invest all the budget in the stronger single mechanism?

**What evidence exists in the paper**

The paper provides indirect evidence that combination could be powerful: the difficulty-dependent complementary strengths (Figures 3 right and 7 right), the fact that both mechanisms individually produce ~4× gains over best-of-N (Figures 4 and 8), and the fact that the PRM and revision model are trained on different distributions (base model outputs vs. revision model outputs) with evidence of transfer failure (Figure 15a). This last point suggests that naive combination—using the base-model PRM to score revision model outputs—would not work, and that careful verifier training would be required. But the paper does not test even this naive combination.

**Mitigation status**

The paper acknowledges this gap as future work in Section 8 but does not attempt even a preliminary experiment. The modular structure of the paper—search in Section 5, revisions in Section 6, FLOPs comparison in Section 7—treats them as separate investigations. A full study of combined search + revisions would require training a PRM on revision model outputs, integrating revision context into the search procedure, and sweeping the additional hyperparameters that combination introduces. This is a substantial undertaking that the paper correctly defers to future work, but it means the current results should be understood as independent demonstrations of two mechanisms rather than as components of an integrated system.

---

### 5. Verifier Over-Optimization Limits Scaling and Is Mitigated but Not Solved

**The assumption or constraint**

All search methods depend on the PRM to score candidate solutions and guide the search toward correct answers. The PRM is trained via Monte Carlo rollouts from the base model (Section 5.1, Appendix D) and is therefore an imperfect proxy for actual correctness. The more aggressively the search optimizes against the PRM, the more it risks finding solutions that score highly under the PRM but are factually incorrect—a phenomenon the paper documents as "over-optimization." The compute-optimal policy mitigates this by routing easy problems away from aggressive search (using best-of-N instead of beam search), but it does not eliminate the underlying problem.

**The consequence**

On medium-difficulty problems where beam search is deployed, over-optimization still limits the scaling ceiling. In Figure 3 (right), the beam search curves flatten or decline at high budgets even in Bins 3–4 where beam search outperforms best-of-N. Lookahead search—the most powerful optimizer because it uses simulated rollouts to improve step-level scoring—paradoxically performs *worst* overall at a given budget (Figure 3, left), because its extra compute cost reduces effective exploration and its improved scoring may be more susceptible to over-optimization. Qualitative examples in Appendix M (Figures 29, etc.) show search producing degenerate outputs: repetitive low-information steps at the end of solutions, overly short 1–2 step solutions that score well under the PRM but are wrong. This means the PRM's reliability is the **true bottleneck** for test-time compute scaling—not the sophistication of the search algorithm, not the compute budget, but how well the verifier distinguishes correct reasoning from plausible-sounding but wrong reasoning under adversarial optimization pressure.

**What evidence exists in the paper**

The evidence is concentrated in Figure 3 (right), where beam search on easy problems (Bin 1) *degrades* with increasing budget—the clearest signature of over-optimization. On Bin 2, the gap between beam search and best-of-N narrows at high budgets rather than widening, suggesting that beam search's advantage saturates. The poor performance of lookahead search (Figure 3, left) despite being a more accurate optimizer further supports the interpretation that better optimization of an imperfect verifier is counterproductive. Appendix M provides qualitative examples of degenerate outputs that score well under the PRM.

**Mitigation status**

The compute-optimal policy is essentially a routing strategy that keeps the optimization budget *below* the over-optimization threshold per difficulty level: use weak optimization (best-of-N) where the verifier is reliable (easy problems) and stronger optimization (beam search) only where the verifier has more room to provide genuine signal (medium problems). This mitigates the symptom but does not solve the underlying problem—improving the PRM itself. The paper does not explore verifier improvements such as adversarial training, ensemble methods, or KL-penalized search that might push the over-optimization threshold higher. Section 8 identifies "verifier over-optimization as the primary bottleneck" and suggests improving verifier robustness as important future work, but does not attempt it within the current study.

---

### 6. Single Benchmark, Single Model Family Constrains Generalizability of All Findings

**The assumption or constraint**

All experiments in the paper use the MATH benchmark (Hendrycks et al., 2021, 500 test questions) with PaLM 2-S* as the base model. The paper argues this model is "representative of the capabilities of many contemporary LLMs" (Section 4) and that MATH is appropriate because test-time compute should help most when the model has the necessary knowledge and the challenge is complex reasoning. However, no experiments are conducted on other reasoning benchmarks (e.g., GSM8k, BBH, ARC), other domains (e.g., code generation, scientific QA, logical reasoning), or other model families (e.g., LLAMA-2, GPT-3.5, MISTRAL).

**The consequence**

Several of the paper's core findings could be specific to MATH, to PaLM 2-S*, or to their interaction. The five-quintile difficulty binning depends on the base model's pass@1 distribution, which would differ for a different base model—a problem that is in Bin 3 (medium) for PaLM 2-S* might be in Bin 1 (easy) for a stronger model or Bin 5 (hard) for a weaker one. The compute-optimal strategy selected per bin (beam search for Bin 3, best-of-N for Bin 1) might not transfer. The PRM's quality and over-optimization behavior depend on PaLM 2-S*'s output distribution and error patterns. The revision model's ability to learn from incorrect in-context examples depends on PaLM 2-S*'s in-context learning capabilities, which vary substantially across model families. The FLOPs-matched comparison uses PaLM 2's specific pretraining compute vs. inference compute ratio, and the 14× larger model's performance is specific to that model family. Nothing in the paper establishes that the central structural findings—adaptive allocation yields 4× gains, difficulty-dependence governs strategy choice, revisions help easy problems while search helps medium ones—would replicate under a different base model or benchmark.

The test set of 500 questions, split into five difficulty quintiles (~100 each), then split by two-fold cross-validation (~50 per fold per bin), means the compute-optimal policy is selected based on very small samples. The paper does not report confidence intervals on the compute-optimal scaling curves, making it impossible to assess whether the observed 4× gains are statistically reliable or could be due to variance in the small per-bin samples. The curves in Figures 4 and 8 are smooth, suggesting genuine trends, but the absence of error bars or statistical testing is a limitation.

**What evidence exists in the paper**

The paper provides no cross-model or cross-benchmark validation. The entire experimental section (Sections 5–7) uses the same base model, the same benchmark, and the same test set. The appendix provides additional qualitative examples and hyperparameter details but no out-of-distribution evaluation. The paper's claims are explicitly scoped to this setting—it does not claim generality—but the absence of any robustness check across domains or models means the findings should be treated as a detailed case study rather than established general principles.

**Mitigation status**

Not addressed. The paper does not discuss the generalizability of findings beyond PaLM 2-S* and MATH. The authors' statement that PaLM 2-S* is "representative" is asserted but not defended with evidence. Future work would need to replicate the study on at least one additional benchmark and one additional model family to establish which findings are robust and which are specific to the MATH-PaLM 2 combination. The release of code and data (Section 8) partially mitigates this by enabling others to run the same experiments on different models and benchmarks, but the paper itself provides no evidence beyond its specific setting.

## 7. Implications and Future Directions
- Field impact
  - Provides a reproducible, open pipeline showing that simple, offline preference optimization (DPO) scales to 70B and meaningfully improves GPT‑4‑judged open‑ended quality (Tables 3–4). This lowers the barrier to building strong open assistants without PPO‑style RL.
  - Offers an improved, public instruction mixture and long‑context training recipe, enabling the community to study how distilled data and context length drive performance (Section 2; Figure 1).
  - Clarifies that parameter‑efficient finetuning may not match full SFT on open‑ended tasks, guiding practitioners’ compute/budget decisions (Table 5).

- Suggested follow‑ups (some named in the Conclusion)
  - Multilingual alignment: Add multilingual SFT and multilingual preference data to recover TyDiQA and test DPO’s behavior in non‑English settings (Section 3.3).
  - RLHF method comparisons at scale: Head‑to‑head 70B‑scale comparisons of DPO vs PPO, rejection sampling (RS/ReST), and offline RL variants, including effects on refusals, verbosity, and factuality (Conclusion).
  - Data ablations: Systematically vary the proportions of distilled vs human‑written data, CoT density, and conversation length to quantify which ingredients drive which metrics (Sections 2–3.2).
  - Length/verbosity control: Incorporate length‑aware preference models or penalties to retain DPO gains while avoiding excessive verbosity (Table 4).
  - Domain‑adaptive assistants: Explore mixtures that retain general ability while selectively leveraging domain‑specialized bases (e.g., hybrid LLAMA‑2/Code LLaMA training or multi‑adapter routing; Table 6 trade‑offs).
  - Larger and newer bases: Apply the recipe to newer base models (e.g., Mistral‑family or successors) and extend beyond 70B with long‑context SFT/DPO.

- Practical applications
  - Open, high‑quality chat assistants for research and industry with strong open‑ended generation (TÜLU 2+DPO 70B, AlpacaEval 95.1 in Table 4).
  - Code‑focused copilots based on `CODE TÜLU 2`, which dramatically improves functional correctness on HumanEval (e.g., 82.5 at 34B; Table 6).
  - Educational and scientific assistants leveraging long‑context capabilities and a science‑task subset (Appendix C), within the multilingual limitations noted.

Block‑quoted highlights
- V2 mixture long‑context coverage gain: 
  > “Moving from 2,048 to 8,192 max length means we only truncate 20 (as opposed to 63,900) samples within our V2 mixture” (Section 2; Figure 1).
- DPO effect on open‑ended quality:
  > “TÜLU 2+DPO 70B … AlpacaEval 95.1 vs 86.6 without DPO; MT‑Bench 7.89 vs 7.49” (Tables 3–4).
- QLoRA trade‑off:
  > “7B AlpacaEval 56.1 (QLoRA) vs 73.9 (full); 70B 78.6 vs 86.6” (Table 5).
- Coding specialization:
  > “CODE TÜLU 2 (7B) Codex‑Eval 68.9 vs 36.9 for TÜLU 2; AlpacaEval drops to 58.0 vs 73.9” (Table 6).

Overall, TÜLU 2 provides a transparent, well‑controlled demonstration that careful data curation plus long‑context SFT and scalable DPO yields state‑of‑the‑art open‑weight instruction‑following models, with clear guidance on when parameter‑efficient methods and domain specialization help or hurt.
