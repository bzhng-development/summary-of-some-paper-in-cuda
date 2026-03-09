## 1. Executive Summary

This paper demonstrates that moderately-sized 7B-parameter Large Language Models (LLMs), specifically `LLAMA2-7B`, `BLOOM-7B`, and `VICUNA-7B`, can be adapted via Parameter-Efficient Fine-Tuning (PEFT) to achieve document-level machine translation (DOCMT) performance that surpasses GPT-4-TURBO on specific tasks within the IWSLT2017 dataset across nine language pairs. The study identifies a critical failure mode where up to 98.3% of translations become "off-target" due to error propagation during autoregressive decoding, a problem mitigated by regenerating context at a higher computational cost. Furthermore, the authors establish that base LLMs fine-tuned with structured prompts outperform instruction-tuned variants and conventional encoder-decoder models in out-of-domain generalization on WMT2023 test sets, while requiring only 1% to 10% of available parallel data to reach peak performance depending on the fine-tuning strategy.

## 2. Context and Motivation

### The Gap: Bridging the Divide Between General LLMs and Specialized Translation
The central problem this paper addresses is the performance gap between **Large Language Models (LLMs)** and **specialized supervised encoder-decoder models** in the specific domain of **Document-Level Machine Translation (DOCMT)**.

While LLMs like GPT-4 have demonstrated remarkable general reasoning capabilities, the paper notes that in translation tasks, only the very largest proprietary models (e.g., GPT-3.5-TURBO, GPT-4-TURBO) can match or surpass state-of-the-art specialized systems like **NLLB** (No Language Left Behind). Crucially, even these massive LLMs often underperform when translating **low-resource languages**. This creates a dichotomy:
1.  **General LLMs:** Possess vast world knowledge and fluency but lack the precise, consistent alignment required for high-quality translation, especially without massive scale.
2.  **Specialized MT Models:** Excel at translation due to task-specific training but often lack the broader contextual understanding and generative flexibility of LLMs.

The specific gap this study targets is whether **moderately-sized LLMs** (specifically those with ~7 billion parameters) can be adapted to close this performance divide. The authors challenge the assumption that only massive models are capable of top-tier translation, investigating if task-specific fine-tuning can unlock superior performance in smaller, more efficient backbones for **document-level** tasks, where maintaining context across multiple sentences is critical.

### Why Document-Level Translation Matters
The focus on **Document-Level Machine Translation (DOCMT)** rather than sentence-level translation is theoretically and practically significant.
*   **Contextual Coherence:** Human communication relies on discourse phenomena that span beyond single sentences, such as pronoun resolution (anaphora), lexical cohesion (consistent terminology), and tense agreement. A sentence-level model might translate "He went to the bank" correctly in isolation but fail to maintain the referent of "He" or the specific meaning of "bank" (financial vs. river) established three sentences prior.
*   **Real-World Impact:** Most real-world translation applications involve documents (legal contracts, literary works, technical manuals, subtitles) where consistency and flow are paramount. Errors in discourse phenomena can render a translation confusing or legally inaccurate, even if individual sentences are grammatically correct.

### Limitations of Prior Approaches
Before this work, the landscape of DOCMT was dominated by two distinct approaches, each with notable shortcomings:

1.  **Traditional Encoder-Decoder Architectures:**
    Prior to the LLM era, DOCMT relied on specialized architectures designed to ingest context. These included:
    *   **Document Embeddings:** Compressing the entire document into a single vector (often losing fine-grained detail).
    *   **Multiple Encoders:** Using separate encoders for the current sentence and the context.
    *   **Attention Variations:** Modifying the attention mechanism to focus on specific prior sentences.
    *   **Translation Caches:** Storing previously translated terms to ensure consistency.
    *   *Shortcoming:* While effective, these models are typically trained from scratch or on limited data compared to LLMs. They often struggle with the "long-tail" of linguistic phenomena and lack the general world knowledge that helps disambiguate difficult terms. Furthermore, recent research suggests that simply concatenating context (a common baseline) often yields diminishing returns without sophisticated architectural changes.

2.  **Prompting Massive LLMs:**
    More recently, researchers have attempted to use massive LLMs (like GPT-3.5/4) for DOCMT purely through **prompting strategies**.
    *   *Shortcoming:* As noted in the Introduction, while GPT-4 is strong, it is not universally superior, particularly in low-resource settings. More importantly, relying on prompting alone treats the model as a black box. It does not leverage the potential of **specializing** the model's weights for the specific task of translation. Additionally, prior work (e.g., Wang et al., 2023b) focused almost exclusively on how prompts affect *inference* for giant models, leaving the question of how to effectively *fine-tune* smaller models for this task largely unexplored.

### Positioning Relative to Existing Work
This paper positions itself as a systematic bridge between these two worlds. It diverges from prior work in three key ways:

*   **From Prompting to Fine-Tuning:** Unlike studies that merely prompt GPT-4 with context, this work investigates **Parameter-Efficient Fine-Tuning (PEFT)** (specifically LoRA) and **Full Fine-Tuning (FFT)** on moderately-sized models (`LLAMA2-7B`, `BLOOM-7B`, `VICUNA-7B`). The hypothesis is that a smaller model, when explicitly trained on document structures, can outperform a larger, general-purpose model that is only prompted.
*   **From Sentence to Document Adaptation:** While recent works have explored LLMs for *sentence-level* translation, this study explicitly targets the **document-level** challenge. It adopts a **two-stage training strategy** (fine-tuning first on monolingual documents to adapt to the target language style, then on parallel documents for translation), a nuance often missed in standard instruction tuning.
*   **Critical Analysis of Failure Modes:** Rather than simply reporting average scores, the paper positions itself to deeply analyze *why* LLMs fail in DOCMT. It specifically investigates **off-target translations** (where the model outputs text in the wrong language) caused by **error propagation** in autoregressive decoding—a phenomenon where an early mistake in a document context corrupts all subsequent sentences. This diagnostic approach provides a more granular understanding of LLM limitations than previous broad evaluations.

In essence, the paper argues that the future of high-quality, efficient machine translation lies not in scaling models indefinitely, but in **adapting** accessible, mid-sized LLMs with rigorous, document-aware training strategies.

## 3. Technical Approach

This section details the methodology used to adapt moderately-sized Large Language Models (LLMs) for high-performance Document-Level Machine Translation (DOCMT). The core idea is that a generic 7-billion parameter LLM, when subjected to a specific two-stage fine-tuning regimen and structured prompting, can outperform both massive proprietary models and specialized encoder-decoder systems on document-level tasks.

### 3.1 Reader orientation (approachable technical breakdown)
The system being built is a specialized translation engine that takes an entire document as input and generates a coherent translation by explicitly conditioning each new sentence on the previously translated context. It solves the problem of "context loss" in standard translation models by treating the translation of a document not as a series of isolated sentences, but as a continuous, autoregressive generation process where the model's own prior outputs serve as the memory for the next step.

### 3.2 Big-picture architecture (diagram in words)
The architecture functions as a sequential pipeline with three primary logical components:
1.  **Context Constructor:** This component takes a raw document, splits it into sentences, and for every target sentence, aggregates the three preceding source-target sentence pairs to form a "context window."
2.  **Prompt Formatter:** This module injects the context window and the current sentence into a specific text template (Prompt 4) that includes explicit natural language instructions, transforming the raw data into a format the LLM can process as a completion task.
3.  **Autoregressive Decoder:** The fine-tuned LLM processes the formatted prompt and generates the translation for the current sentence; crucially, this generated output is then fed back into the Context Constructor to serve as part of the context for the *next* sentence in the sequence.

### 3.3 Roadmap for the deep dive
To fully understand the mechanics of this approach, we will proceed in the following order:
*   **Training Strategy:** We first explain the unique two-stage fine-tuning process (Monolingual $\rightarrow$ Parallel) that adapts the model's linguistic capabilities before teaching it translation, distinguishing it from standard single-stage tuning.
*   **Prompt Engineering & Data Formulation:** We dissect the specific input formats (Prompts 1–4) used during training, explaining how the arrangement of context and instructions influences the model's ability to learn discourse phenomena.
*   **Model Backbones & Fine-Tuning Methods:** We detail the specific models used (`LLAMA2`, `BLOOM`, `VICUNA`) and compare the two adaptation techniques: Full Fine-Tuning (FFT) versus Parameter-Efficient Fine-Tuning (LoRA), including their parameter counts and computational implications.
*   **Inference & Decoding Dynamics:** We describe the step-by-step generation process at test time, specifically highlighting the "REUSE" strategy where model outputs are recycled as context, and the alternative "REGEN" strategy used to diagnose error propagation.
*   **Hyperparameters & Optimization:** We conclude with the precise numerical configurations (learning rates, batch sizes, epochs) that govern the training stability and convergence of these large models.

### 3.4 Detailed, sentence-based technical breakdown

**Core Philosophy and Pipeline Flow**
This is an empirical adaptation study that treats Document-Level Machine Translation as a conditional text generation problem, where the probability of a target sentence depends not only on its source counterpart but also on the history of the document. The data pipeline operates sequentially: first, raw parallel documents from the IWSLT2017 dataset are segmented into sentences; second, for each sentence index $i$ (where $i > 1$), the system constructs a training example comprising the three preceding sentence pairs $(S_{i-3}, T_{i-3}), (S_{i-2}, T_{i-2}), (S_{i-1}, T_{i-1})$ and the current pair $(S_i, T_i)$; third, these elements are serialized into a text string using a specific prompt template; and finally, this string is fed to the LLM, which is trained to minimize the negative log-likelihood of the target tokens.

**Two-Stage Training Strategy**
The authors employ a **two-stage training strategy** to mitigate the risk of catastrophic forgetting and to better align the model with the target language's stylistic norms before introducing the translation task.
*   **Stage 1: Monolingual Fine-Tuning:** In the first stage, all parameters of the base LLM are fine-tuned exclusively on **monolingual documents** from the target language. The objective here is not translation, but rather domain adaptation; by exposing the model to 100 million tokens of target-language text (sampled from the CulturaX corpus using data pruning techniques), the model learns the specific syntactic structures and discourse flows of the target language without the noise of source-language interference. This addresses the limitation that many LLMs (like LLAMA2) are predominantly pre-trained on English-centric data.
*   **Stage 2: Parallel Document Fine-Tuning:** Only after the model has adapted to the target language style does it enter the second stage, where it is fine-tuned on **parallel document corpora**. In this phase, the model learns the mapping between source and target sentences while maintaining the document-level context established in Stage 1. This contrasts with traditional approaches that often start directly with parallel sentence pairs, a method the authors found to be sub-optimal for LLMs in their ablation studies (Table 6).

**Prompt Engineering and Context Formulation**
A critical design choice in this work is the structure of the input prompt, as LLMs are highly sensitive to how context is presented. The study investigates four distinct prompt variations to determine the optimal balance between structural clarity and instructional guidance.
*   **Prompt Structure:** The prompts generally consist of a "Context" section containing previous sentences and a "Current Sentence" section. For example, **Prompt 1** simply lists the source and target context followed by the current source sentence, expecting the model to complete the target. **Prompt 2** structures the context as alternating source-target pairs line-by-line, which the authors found generally outperforms the block-text style of Prompt 1.
*   **Instructional Guidance:** **Prompt 3** and **Prompt 4** introduce explicit natural language instructions, such as "Given the provided parallel context, translate the following..." before the current sentence. The study reveals that while instructions help base models (like LLAMA2), they can be redundant or even slightly detrimental for models already instruction-tuned (like VICUNA), unless combined with the superior structural formatting of Prompt 2.
*   **The Chosen Configuration (Prompt 4):** Based on preliminary results (Table 1), the authors select **Prompt 4** for all main experiments. This prompt combines the line-by-line alternating structure of Prompt 2 with the explicit instruction of Prompt 3. Mathematically, if $C_{src}$ and $C_{tgt}$ represent the concatenated sequences of the previous $k=3$ source and target sentences, and $s_i$ is the current source sentence, the input $X$ to the model is formatted as:
    $$ X = \text{Format}(C_{src}, C_{tgt}) \oplus \text{Instruction} \oplus s_i $$
    where $\oplus$ denotes string concatenation and $\text{Format}$ arranges the context as alternating pairs. The model is then trained to maximize the likelihood of the target sentence $t_i$ given $X$.

**Model Backbones and Fine-Tuning Mechanisms**
The study utilizes three distinct 7-billion parameter backbones to evaluate the impact of pre-training data and instruction tuning on DOCMT performance.
*   **LLAMA2-7B:** A base model pre-trained primarily on English data, serving as a control for English-centric knowledge.
*   **BLOOM-7B:** A model pre-trained on a diverse mix of 46 languages, hypothesized to have better inherent multilingual capabilities.
*   **VICUNA-7B:** An instruction-tuned variant of LLAMA2, used to test whether prior alignment with human instructions aids or hinders task-specific translation fine-tuning.

The authors compare two fine-tuning methodologies for adapting these backbones:
1.  **Full Fine-Tuning (FFT):** This approach updates **all** parameters of the 7B model. While computationally expensive, it allows the model to completely reshape its internal representations for the translation task. In the notation of the paper, these models are denoted as `L-7B-FFT`, `B-7B-FFT`, and `V-7B-FFT`.
2.  **Parameter-Efficient Fine-Tuning (PEFT) via LoRA:** This method freezes the original model weights and injects trainable low-rank decomposition matrices into the transformer layers. Specifically, the authors use a **LoRA rank of 16**, which affects only approximately **8 million parameters** (roughly 0.1% of the total 7 billion). This drastically reduces memory usage and training time. These models are denoted as `L-7B-LORA`, `B-7B-LORA`, and `V-7B-LORA`. The results indicate that LoRA often outperforms FFT on the full dataset, likely because updating fewer parameters prevents overfitting on the relatively small parallel document corpus.

**Inference and Decoding Dynamics**
The inference process in document-level translation introduces a unique challenge known as **error propagation**, which the paper analyzes in depth.
*   **The REUSE Strategy:** During standard inference (denoted as **REUSE**), the model translates the document sentence by sentence. For the first sentence, there is no context. Once the model generates the translation for sentence 1 ($\hat{t}_1$), this generated hypothesis is *reused* as the ground-truth context for translating sentence 2. This continues sequentially: the context for sentence $i$ includes the model's own previous outputs $\hat{t}_{i-3}, \dots, \hat{t}_{i-1}$. If the model makes an error in $\hat{t}_1$ (e.g., hallucinating a word or translating into the wrong language), this error becomes part of the context for $\hat{t}_2$, potentially causing the model to drift further off-target. This autoregressive feedback loop is identified as the primary cause of the high "off-target" translation rates observed in some configurations.
*   **The REGEN Strategy:** To isolate and verify the impact of error propagation, the authors propose an alternative decoding strategy called **REGEN**. In this mode, instead of reusing the model's potentially flawed hypotheses, the system re-generates the context translations from scratch or uses gold references (in experimental settings) to break the chain of errors. While Table 4 shows that REGEN significantly improves translation quality (e.g., raising sBLEU from 3.9 to 17.5 for Arabic-English), it comes at a prohibitive computational cost, requiring roughly **4 times** more inference steps than REUSE, as the model must re-process the entire context history for every single sentence.

**Hyperparameters and Optimization Details**
The reproducibility of these results relies on specific hyperparameter configurations detailed in Appendix B.
*   **Monolingual Stage:** The models are fine-tuned with a learning rate of $5 \times 10^{-5}$ and a large batch size of **256**. A linear learning rate schedule is used with a warm-up phase covering the first **10%** of training steps.
*   **Parallel Stage:** For the LoRA models, the learning rate remains $5 \times 10^{-5}$, but the batch size is reduced to **64** due to memory constraints of the adapter layers. The models are trained for up to **3 epochs** with early stopping based on validation loss. In contrast, the baseline MT5 models are trained for **10 epochs** with a higher learning rate of $5 \times 10^{-4}$.
*   **Decoding Settings:** During evaluation, the models use **beam search** with a beam size of **5** to explore multiple potential translations and select the most probable sequence, a standard practice to improve fluency over greedy decoding.

**Design Choices and Rationale**
The decision to use a **context window of three preceding sentences** is based on prior work (Wang et al., 2023a) suggesting that this span captures sufficient discourse information (such as pronoun antecedents) without overwhelming the model's context window or introducing irrelevant noise. Furthermore, the choice to fine-tune on **monolingual data first** is a strategic move to decouple language adaptation from translation learning; by first ensuring the model is fluent in the target language's document style, the subsequent parallel fine-tuning can focus purely on alignment, leading to more stable convergence. Finally, the preference for **base models** (LLAMA2) over **instruction-tuned models** (VICUNA) for the backbone is driven by the finding that instruction tuning can sometimes constrain the model's flexibility, whereas base models, when given structured prompts (like Prompt 4), demonstrate superior zero-shot cross-lingual transfer and adaptability to the specific DOCMT format.

## 4. Key Insights and Innovations

This study moves beyond simply reporting that "LLMs can translate" to uncovering specific, often counter-intuitive mechanisms that govern their success and failure in document-level tasks. The following insights represent fundamental shifts in how we understand the adaptation of large models for translation, distinguishing between mere performance tweaks and structural discoveries about model behavior.

### 4.1 The Paradox of Off-Target Translation via Error Propagation
**The Innovation:** The most critical discovery of this paper is the identification of **autoregressive error propagation** as the primary cause of catastrophic failure in LLM-based DOCMT, manifesting as "off-target translation" (generating text in a language other than the target).

*   **Distinction from Prior Work:** Previous research on machine translation errors typically focused on semantic inaccuracies (mistranslations) or grammatical faults within the *correct* target language. Standard encoder-decoder models rarely suffer from "language drift" where the output suddenly switches to French when translating German to English. This paper reveals that LLMs, due to their generative nature, are uniquely susceptible to a feedback loop: if the model makes a slight error in the first sentence of a document (e.g., mixing in source language tokens), that error becomes part of the *context* for the second sentence. The model then interprets this mixed-language context as a signal to continue generating in that mixed style, leading to a cascade where up to **98.3%** of the document becomes off-target (Table 3, Figure 2).
*   **Significance:** This finding fundamentally changes the evaluation of LLM reliability. It demonstrates that high average scores (like COMET) can mask total failures on specific instances. The authors prove this mechanism by introducing the **REGEN** decoding strategy (Table 4). By forcing the model to re-generate or use gold context for every step—breaking the error chain—performance on difficult pairs like Arabic-English jumps from an sBLEU of **3.9** to **17.5**. This confirms that the model *knows* how to translate; it is simply being misled by its own previous outputs. This is a theoretical advance in understanding the stability limits of autoregressive generation in long-context tasks.

### 4.2 The Superiority of Base Models Over Instruction-Tuned Variants
**The Innovation:** The study provides empirical evidence that **base pre-trained models** (e.g., `LLAMA2-7B`) significantly outperform their **instruction-tuned counterparts** (e.g., `VICUNA-7B`) when adapted for specialized document-level translation via supervised fine-tuning.

*   **Distinction from Prior Work:** The prevailing assumption in the LLM era is that instruction tuning (aligning models to follow human commands) universally enhances performance on downstream tasks. Consequently, many practitioners default to chat-oriented models for all applications. This paper challenges that dogma in the specific domain of DOCMT. The results show that instruction-tuned backbones (`V-7B`) often exhibit **negative zero-shot cross-lingual transfer** (Table 8), with COMET score differences ($\Delta$) dropping as low as **-34.1** for Chinese, whereas base models (`L-7B`) show consistent positive gains (e.g., **+37.2** for Chinese).
*   **Significance:** This suggests that instruction tuning may inadvertently constrain the model's latent multilingual capabilities or bias it towards conversational patterns that conflict with the rigid structural requirements of document translation. The base models, when guided by the structured **Prompt 4** (described in Section 3), retain a broader flexibility that allows them to learn the translation mapping more effectively. This is a crucial design choice for practitioners: for high-stakes, structured tasks like translation, "raw" base models fine-tuned with explicit prompts are superior to "aligned" chat models.

### 4.3 Divergent Data Efficiency Scaling Laws: LoRA vs. Full Fine-Tuning
**The Innovation:** The paper establishes distinct **scaling laws** for data efficiency depending on the fine-tuning method, revealing that Full Fine-Tuning (FFT) is vastly more data-efficient than Parameter-Efficient Fine-Tuning (LoRA) in the low-data regime, despite LoRA's overall superior performance on full datasets.

*   **Distinction from Prior Work:** LoRA is widely celebrated for enabling fine-tuning on consumer hardware with minimal data. However, this study nuances that view by plotting performance against the percentage of training data (Figure 4). It reveals a crossover point: **FFT models reach near-peak performance with only ~1% of the dataset** (approx. 2,000 examples), whereas **LoRA models require ~10%** (approx. 20,000 examples) to achieve comparable results.
*   **Significance:** This insight is vital for **low-resource language pairs** where parallel document data is scarce. If a researcher has only a few thousand document pairs, FFT is the theoretically optimal choice, contradicting the general preference for LoRA. Conversely, when data is abundant, LoRA surpasses FFT because updating all 7 billion parameters (FFT) leads to rapid **overfitting**, while LoRA's constrained updates (8M parameters) act as a regularizer. This defines a clear boundary condition for method selection based on data availability, moving beyond a "one-size-fits-all" approach to adaptation.

### 4.4 Out-of-Domain Generalization vs. In-Domain Specialization
**The Innovation:** The study uncovers a trade-off where LLM-based DOCMT models, while sometimes trailing specialized encoder-decoder models (like `DOC2DOC-MT5`) on in-domain benchmarks (IWSLT2017), demonstrate **superior generalization** to out-of-domain text (WMT2023).

*   **Distinction from Prior Work:** Traditional metrics often prioritize in-domain performance, leading to the conclusion that specialized small models are "better" than adapted LLMs. Table 2 shows `DOC2DOC-MT5` beating L-7B-LORA on IWSLT2017. However, when evaluated on the fresh, unseen **WMT2023 test sets** (Table 7), the trend reverses: `L-7B-LORA` achieves a dBLEU of **28.9** and COMET of **76.4**, significantly outperforming the specialized MT5 baseline (dBLEU **20.2**, COMET **74.4**).
*   **Significance:** This indicates that the broad world knowledge encoded in the 7B pre-training allows the model to handle vocabulary and domains it has never seen during fine-tuning, whereas specialized models like MT5 overfit to the specific domain of their training data (TED talks). This validates the hypothesis that LLMs offer a more robust foundation for real-world translation scenarios where test data rarely matches the training distribution perfectly. It shifts the metric of success from "peak in-domain accuracy" to "robustness across domains."

### 4.5 The Compound Effect of Structured Context and Explicit Instructions
**The Innovation:** Through a systematic ablation of prompt designs, the paper identifies that the optimal prompt for DOCMT is not just about providing context, but about the **synergistic combination** of specific structural formatting and explicit natural language instructions.

*   **Distinction from Prior Work:** Prior prompting studies often treat "context" and "instructions" as independent variables or focus solely on few-shot examples. This work (Section 4, Table 1) dissects the interaction. It finds that while structured context (alternating source-target pairs, **Prompt 2**) is generally better than block text (**Prompt 1**), and instructions (**Prompt 3**) help base models, the **combination (Prompt 4)** yields the highest performance across all backbones. Notably, it highlights a subtle interaction: instructions are less effective for already instruction-tuned models (VICUNA) unless the context structure is also optimized.
*   **Significance:** This provides a prescriptive recipe for adapting LLMs to structured generation tasks. It proves that the *format* in which context is presented (line-by-line alignment) is as critical as the *presence* of context. This "compound effect" suggests that LLMs rely on both semantic understanding (driven by instructions) and pattern recognition (driven by structural formatting) to correctly resolve discourse phenomena like pronoun antecedents.

## 5. Experimental Analysis

This section dissects the empirical evidence provided in the paper, moving from the setup of the evaluation to a granular analysis of the results. The authors do not merely report average scores; they construct a multi-faceted evaluation framework designed to stress-test the models on discourse coherence, data efficiency, and robustness to distribution shifts. The analysis reveals a complex landscape where adapted 7B models can surpass GPT-4 in specific niches while suffering from catastrophic failure modes in others.

### 5.1 Evaluation Methodology and Baselines

To rigorously assess Document-Level Machine Translation (DOCMT), the study employs a comprehensive setup involving diverse datasets, multiple metrics, and a tiered set of baselines ranging from specialized small models to massive proprietary LLMs.

**Datasets: In-Domain vs. Out-of-Domain**
The primary evaluation ground is the **IWSLT2017** dataset, consisting of TED talk transcripts. This dataset covers **nine language pairs** with English (Arabic, German, French, Italian, Japanese, Korean, Dutch, Romanian, Chinese), totaling approximately **1.9K sentence-aligned parallel documents** and **240K sentences** per pair (Section 3.2). This serves as the "in-domain" benchmark.
Crucially, to test generalization and mitigate data leakage risks inherent in large pre-trained models, the authors introduce a secondary evaluation on the **WMT2023** test sets for English-German translation. This acts as a strict "out-of-domain" check, ensuring that high scores are not merely artifacts of the models having memorized TED talk styles during pre-training.

**Metrics: Beyond BLEU**
Recognizing that standard metrics often fail to capture document-level nuances, the study utilizes a triad of evaluation measures:
*   **sBLEU:** Sentence-level BLEU, measuring local translation accuracy.
*   **dBLEU:** Document-level BLEU, which calculates n-gram matches over the entire document, penalizing inconsistencies in terminology or style across sentences.
*   **COMET:** A neural evaluation metric trained on human judgments, which correlates better with semantic quality than BLEU.
The paper reports the average of these metrics ($\mu$sBLEU, $\mu$dBLEU, $\mu$COMET) across all language pairs to provide a holistic view.

**Baselines: The Competitive Landscape**
The authors compare their adapted LLMs against three distinct tiers of competitors (Section 3.3):
1.  **State-of-the-art Specialized MT:** The **NLLB** (No Language Left Behind) family in sizes 600M, 1.3B, and 3.3B parameters, plus **Google Translate**. These represent the ceiling for dedicated translation systems.
2.  **Massive Proprietary LLMs:** **GPT-3.5-TURBO** and **GPT-4-TURBO**, prompted with the optimal strategy (Prompt 4). These represent the "gold standard" of general intelligence.
3.  **Re-implemented DOCMT Baselines:** Strong encoder-decoder models like **DOC2DOC-MT5** (up to 1.2B parameters) and recent architectures like **MR-DOC2SEN** and **DOCFLAT**. These serve as the direct competitors for the document-level task.

**The Adapted Models**
The core experimental subjects are three 7B-parameter backbones adapted via two methods:
*   **Backbones:** `LLAMA2-7B` (English-centric base), `BLOOM-7B` (multilingual base), and `VICUNA-7B` (instruction-tuned).
*   **Adaptation:** **Full Fine-Tuning (FFT)** updating all 7B parameters, and **LoRA** (Parameter-Efficient Fine-Tuning) updating only ~8M parameters (rank 16).
This yields six primary model variants (e.g., `L-7B-LORA`, `L-7B-FFT`).

### 5.2 Main Quantitative Results: Selective Excellence

The headline finding of the paper is that moderately-sized, adapted LLMs can outperform GPT-4 on specific tasks, but this superiority is highly conditional on the translation direction and the metric used.

**Overall Performance on IWSLT2017**
Table 2 presents the aggregate results. A striking asymmetry emerges between translating *from* English (En-X) and *to* English (X-En):

*   **English-to-Other (En-X):** Specialized models still dominate. The 3.3B parameter **NLLB** achieves a $\mu$dBLEU of **30.5**, and **GPT-4-TURBO** reaches **30.7**. In contrast, the best adapted LLM (`L-7B-LORA`) scores only **20.2**. Here, the 7B models struggle to match the precision of dedicated systems or the massive scale of GPT-4.
*   **Other-to-English (X-En):** The narrative flips. The adapted LLMs shine. `B-7B-LORA` (based on the multilingual BLOOM backbone) achieves a $\mu$sBLEU of **29.9** and $\mu$dBLEU of **33.6**, surpassing **GPT-4-TURBO** (31.7 sBLEU / 35.1 dBLEU) in sentence-level metrics and coming very close in document-level scores. More importantly, in terms of **COMET** (semantic quality), `B-7B-LORA` hits **81.4**, which is competitive with GPT-4's **86.0** given the massive parameter disparity (7B vs. likely >100B).

**The "Selective Excellence" Phenomenon**
Figure 2 provides a breakdown by language pair for X-En tasks, revealing extreme variance.
*   **Success Cases:** For **German-to-English (De-En)** and **French-to-English (Fr-En)**, the adapted LLMs (`L-7B-LORA`, `B-7B-LORA`) often match or exceed GPT-4-TURBO. For instance, in COMET scores for German-English, `L-7B-LORA` reaches levels comparable to the giant models.
*   **Failure Cases:** Conversely, for **Arabic-to-English (Ar-En)** and **Chinese-to-English (Zh-En)**, some models collapse entirely. `L-7B-LORA` scores near zero on certain metrics for these pairs.
*   **Interpretation:** This suggests that the pre-training data distribution of the backbone heavily influences success. `BLOOM-7B`, with its multilingual pre-training, consistently outperforms `LLAMA2-7B` on non-English source languages, confirming that base model choice is a critical hyperparameter for DOCMT.

**Fine-Tuning Strategy: LoRA vs. FFT**
Table 2 also highlights a nuanced trade-off between Full Fine-Tuning (FFT) and LoRA:
*   **General Trend:** LoRA models generally outperform FFT models on the full dataset. For example, `L-7B-LORA` (20.2 dBLEU En-De) beats `L-7B-FFT` (16.2 dBLEU).
*   **The Overfitting Hypothesis:** The authors attribute this to overfitting. FFT updates all 7 billion parameters, which allows the model to rapidly memorize the relatively small IWSLT training set (~240K sentences), harming generalization. LoRA, by constraining updates to only **0.1%** of parameters (~8M), acts as a strong regularizer, preserving the model's general knowledge while adapting it to the task.
*   **Exception:** In the X-En direction, `V-7B-FFT` (Vicuna fully fine-tuned) occasionally surpasses its LoRA counterpart, suggesting that for specific high-resource pairs flowing into English, the capacity of full fine-tuning can be beneficial if overfitting is managed.

### 5.3 Critical Failure Analysis: The Off-Target Catastrophe

The most significant contribution of this experimental analysis is the identification and quantification of **off-target translation**, a failure mode where the model generates text in a language other than the target (e.g., outputting source language or random code-switching).

**Quantifying the Failure**
Table 3 reveals alarming off-target rates for specific configurations.
*   **Severity:** For the **Korean-to-English** task, `V-7B-FFT` exhibits an off-target rate of **98.3%**. Similarly, `L-7B-FFT` fails on **Japanese-to-English** with an **87.9%** off-target rate.
*   **Model Dependency:** The failure is not uniform. `B-7B-LORA` (BLOOM backbone) maintains consistently low off-target rates (e.g., **1.6%** for Zh-En, **4.0%** for Ko-En), underscoring the advantage of multilingual pre-training. In contrast, English-centric backbones like LLAMA2 and instruction-tuned Vicuna are highly prone to this drift when fine-tuned aggressively (FFT).

**Diagnosing the Cause: Error Propagation**
The authors hypothesize that this is caused by **error propagation** in the autoregressive decoding process. In DOCMT, the model's own previous output becomes the context for the next sentence. If the model makes a small error in sentence 1 (e.g., slipping into the source language), sentence 2 sees that error as context and is more likely to repeat it, creating a positive feedback loop of degradation.

**The REUSE vs. REGEN Experiment**
To prove this, Table 4 presents a controlled experiment on Arabic-English translation comparing two decoding strategies:
*   **REUSE:** The standard approach where the model's generated hypothesis is reused as context. Result: `L-7B-LORA` achieves an sBLEU of only **3.9**.
*   **REGEN:** An oracle-like strategy where the context is re-generated or corrected at every step (breaking the error chain). Result: sBLEU jumps to **17.5**.
*   **Conclusion:** The massive gap (3.9 vs. 17.5) confirms that the model *possesses* the translation capability but is crippled by its own accumulating errors during inference. The cost of fixing this via REGEN is high: it requires **4x** the computational inference time, making it impractical for real-world deployment.

### 5.4 Ablation Studies and Robustness Checks

The paper includes several rigorous ablation studies that validate the design choices and reveal deeper insights into model behavior.

**Training Strategy Ablation (Table 6)**
The authors test the necessity of their **two-stage training** (Monolingual $\rightarrow$ Parallel) against:
*   **One-Stage:** Direct fine-tuning on parallel documents.
*   **Three-Stage:** Adding an extra parallel *sentence* fine-tuning step.
Results show that for LLMs, the **Two-Stage** approach is optimal. For high-performing languages like Dutch-English, Two-Stage yields a COMET of **87.0**, while One-Stage lags at **71.2**. For low-resource languages like Arabic, the gap is even starker (51.6 vs. 50.1). This confirms that adapting the model to the target language's *style* (via monolingual data) before teaching it *translation* is crucial for stability.

**Data Efficiency Scaling Laws (Figure 4)**
Figure 4 plots COMET scores against the percentage of training data used, revealing distinct scaling behaviors:
*   **FFT Efficiency:** Fully Fine-Tuned models are incredibly data-efficient. They reach near-peak performance with only **1%** of the dataset (~2,000 examples).
*   **LoRA Data Hunger:** LoRA models require **10%** of the dataset (~20,000 examples) to match the performance of FFT at 1%.
*   **Implication:** This creates a clear decision boundary for practitioners. If data is scarce (low-resource languages), **FFT** is superior despite the overfitting risk on large datasets. If data is abundant, **LoRA** is preferred for its better final performance and lower compute cost.

**Out-of-Domain Generalization (Table 7)**
On the **WMT2023** test sets (fresh data), the adapted LLMs demonstrate superior robustness compared to specialized models.
*   **LLM Advantage:** `L-7B-LORA` achieves a dBLEU of **28.9** and COMET of **76.4** on En-De.
*   **Specialized Model Struggle:** The best MT5 baseline (`IADA-MT5`) scores only **21.2** dBLEU and **75.4** COMET.
*   **Significance:** While specialized models won on the in-domain IWSLT data (Table 2), they overfit to the TED talk domain. The LLMs, leveraging their broad pre-training, generalize significantly better to unseen domains, validating their utility for real-world applications where test data distributions shift.

**Discourse Phenomena (Table 5)**
Using contrastive test sets designed to evaluate pronoun resolution (e.g., choosing the correct gender for "he/she" based on context), the study finds:
*   `L-7B-LORA` achieves **83.1%** accuracy on En-De, outperforming the specialized `DOC2DOC-MT5` (77.0%).
*   However, `B-7B` models perform worse (**75.5%**), which the authors attribute to BLOOM's lack of German text in its pre-training. This reinforces that **contextual understanding in DOCMT is largely inherited from pre-training**, not just learned during fine-tuning.

### 5.5 Assessment of Claims and Limitations

Do the experiments support the claims?
*   **Claim:** *Moderately-sized LLMs can surpass GPT-4.* **Supported, but conditionally.** This holds true for specific X-En tasks (Table 2, Figure 2) and out-of-domain generalization (Table 7), but fails for En-X tasks where GPT-4 and NLLB remain superior.
*   **Claim:** *Off-target translation is due to error propagation.* **Strongly Supported.** The REUSE vs. REGEN experiment (Table 4) provides causal evidence, showing a 4x improvement when the error chain is broken.
*   **Claim:** *Base models outperform instruction-tuned models.* **Supported.** Table 8 shows `L-7B` (base) having positive zero-shot transfer (+29.4 avg $\Delta$), while `V-7B` (instruction-tuned) suffers negative transfer (-8.9 avg $\Delta$).

**Mixed Results and Trade-offs**
The results are not a universal endorsement of LLMs for translation. The study exposes a sharp trade-off:
1.  **Stability vs. Capacity:** FFT offers high data efficiency but risks catastrophic overfitting and off-target drift (Table 3). LoRA is stable and performs well on full data but requires more examples to converge (Figure 4).
2.  **Quality vs. Cost:** The "fix" for off-target errors (REGEN) improves quality drastically but increases inference cost by **400%**, rendering it impractical for production without architectural changes.
3.  **Domain Specificity:** While LLMs generalize better to new domains, they currently lag behind specialized models on in-domain precision for English-to-Other translation tasks.

In summary, the experimental analysis paints a picture of a technology in transition. The 7B models are powerful but fragile. They possess the *potential* for state-of-the-art translation, as evidenced by their out-of-domain robustness and high scores in favorable conditions, but they are currently held back by autoregressive instability and sensitivity to pre-training data distributions.

## 6. Limitations and Trade-offs

While this study demonstrates that moderately-sized LLMs can achieve competitive document-level translation, the authors explicitly acknowledge significant constraints that prevent these models from being a universal solution. The approach relies on specific assumptions about data availability and model stability, and it introduces trade-offs between translation quality, computational cost, and robustness that must be carefully managed.

### 6.1 Constraints on Model Scale and Generalizability
The most fundamental limitation of this work is its confinement to **moderately-sized models (7 billion parameters)**.
*   **The Assumption:** The study operates under the hypothesis that task-specific fine-tuning can compensate for the lack of scale inherent in 7B models compared to giants like GPT-4.
*   **The Constraint:** As stated in Section 8 ("Constraints on Model Scale"), the authors explicitly note that their findings **might vary if conducted with larger models**. The observed behaviors—such as the severity of off-target translation or the specific data efficiency curves—may not hold for 13B, 70B, or larger architectures.
*   **Implication:** We cannot definitively conclude that the "sweet spot" for efficient DOCMT is at 7B parameters. It remains an open question whether scaling up the backbone while applying the same two-stage fine-tuning strategy would eliminate the catastrophic failure modes (like the 98.3% off-target rate) observed in smaller models, or if those errors are intrinsic to the autoregressive decoding mechanism regardless of scale.

### 6.2 The Instability of Supervised Fine-Tuning
A critical weakness identified is the **instability** of the training process for LLM-based DOCMT.
*   **Evidence of Instability:** In Section 8 ("Instability in Training"), the authors report "noticeable inconsistencies in performance" that exceed standard training randomness. Figure 4 illustrates this volatility, where performance curves for different fine-tuning strategies do not always converge smoothly.
*   **Non-Convergence:** In some experimental runs, the fine-tuning process **failed to reach convergence** entirely. The authors attribute this to the complexity of adapting a general-purpose language model to a rigid, structured translation task with limited parallel document data.
*   **Resource Barrier to Resolution:** Crucially, the paper admits that due to **limited computational resources**, the authors could not perform an in-depth investigation into these failures or devise robust remedies (e.g., advanced regularization techniques or alternative optimizers). This leaves a significant gap in understanding *why* certain training runs fail and how to guarantee stability in production environments.

### 6.3 The Quality-Cost Trade-off: The REGEN Dilemma
The study identifies a stark trade-off between translation quality and inference cost, centered on the **error propagation** problem.
*   **The Problem:** As detailed in Section 6 and Table 4, the standard decoding strategy (**REUSE**), where the model's own output serves as context for the next sentence, leads to catastrophic error propagation. For difficult pairs like Arabic-English, this results in an sBLEU of only **3.9**.
*   **The "Fix" and Its Cost:** The alternative strategy, **REGEN**, which regenerates or corrects context at every step to break the error chain, boosts performance significantly (sBLEU jumps to **17.5**). However, the authors quantify the cost: REGEN requires approximately **4 times** the computational inference steps compared to REUSE.
*   **Practical Implication:** This creates a prohibitive barrier for real-world deployment. While the model *can* translate well if given infinite compute to re-verify context, the **400% increase in latency and cost** makes the high-quality mode impractical for large-scale applications. The paper offers no architectural solution to mitigate error propagation without this massive computational penalty.

### 6.4 Sensitivity to Prompting and Sub-Optimal Configurations
The performance of these adapted models is heavily dependent on the specific **prompting strategy** used during fine-tuning, introducing a fragility not present in traditional encoder-decoder models.
*   **Prompt Dependency:** Section 4 and Table 1 demonstrate that performance varies wildly based on prompt structure (e.g., Prompt 1 vs. Prompt 4). The chosen optimal prompt (Prompt 4) combines specific structural formatting with natural language instructions.
*   **The Limitation:** In Section 8 ("Influence of Prompting Techniques"), the authors concede that the recommended prompt **may not be the most effective possible**. Since they only tested four variations, there is a risk that the reported results are sub-optimal due to imperfect prompt engineering rather than model limitations.
*   **Operational Risk:** This sensitivity implies that deploying these models requires careful, task-specific prompt tuning. A slight deviation in input format could degrade performance significantly, making the system less robust to distribution shifts in input data compared to models that rely purely on learned weights.

### 6.5 Unaddressed Scenarios and Edge Cases
Several important real-world scenarios remain unaddressed or poorly handled by the proposed approach:
*   **Extremely Low-Resource Languages:** While the study covers nine language pairs, it relies on the existence of **parallel document corpora** (even if small) for the second stage of training. The approach does not address languages where *no* parallel documents exist, only sentences or phrases. The two-stage strategy (Monolingual $\rightarrow$ Parallel Documents) breaks down if the parallel document stage cannot be constructed.
*   **Very Long Documents:** The context window is fixed at **three preceding sentences** (Section 3.1). This design choice, while based on prior work, limits the model's ability to capture discourse phenomena that span entire chapters or long technical manuals. The paper does not evaluate how the model performs when the relevant context lies beyond this three-sentence horizon.
*   **Domain Shifts Beyond WMT2023:** Although the study evaluates generalization on WMT2023 test sets, the training data is exclusively from **TED talks** (IWSLT2017). TED talks have a specific style (oral, persuasive, relatively informal). The model's performance on highly formal domains (legal contracts, medical records) or highly specialized technical domains remains an open question, as the "monolingual adaptation" stage may not sufficiently cover these registers.

### 6.6 Summary of Open Questions
The paper leaves several critical questions unanswered for future research:
1.  **Scalability of Stability:** Does increasing the model size beyond 7B parameters naturally resolve the training instability and off-target drift, or are these fundamental flaws in autoregressive DOCMT?
2.  **Efficient Error Mitigation:** Can architectural modifications (e.g., non-autoregressive decoding, explicit memory modules) break the error propagation loop without incurring the **4x cost** of the REGEN strategy?
3.  **Instruction Tuning Nuances:** Why exactly do instruction-tuned backbones (VICUNA) suffer from negative zero-shot transfer in this domain? Is this a flaw in the instruction tuning data itself, or a mismatch with the rigid structure of document translation?

In conclusion, while the paper successfully proves that 7B LLMs can be adapted for high-quality document-level translation, it simultaneously reveals that this capability is **fragile, computationally expensive to stabilize, and highly sensitive to hyperparameters**. The path to robust, production-ready LLM-based DOCMT requires solving the error propagation problem without massive compute overhead and ensuring training stability across diverse model scales.

## 7. Implications and Future Directions

This study fundamentally reshapes the trajectory of machine translation research by shifting the focus from **scaling model size** to **optimizing adaptation strategies**. It demonstrates that the "bigger is better" paradigm has diminishing returns for specific, structured tasks like Document-Level Machine Translation (DOCMT) when compared to rigorous, task-specific fine-tuning of moderately-sized models. The findings suggest that the future of high-quality, efficient translation lies not in waiting for trillion-parameter models, but in developing robust methodologies to adapt accessible 7B-parameter backbones.

### 7.1 Reshaping the Field: From Prompting to Specialized Adaptation
The most profound implication of this work is the validation of **specialized adaptation over general prompting**. Prior to this study, the dominant approach to leveraging LLMs for translation was "prompt engineering" massive proprietary models (e.g., GPT-4). This paper proves that a **7B parameter model**, when subjected to a **two-stage fine-tuning regimen** (Monolingual $\rightarrow$ Parallel) and structured prompting, can outperform GPT-4-TURBO on specific tasks (Table 2, Figure 2) and generalize better to out-of-domain data (Table 7).

This shifts the research landscape in three key ways:
*   **Democratization of State-of-the-Art:** It suggests that top-tier translation capabilities are no longer the exclusive domain of entities with access to massive compute clusters. Researchers and organizations can achieve competitive results using open-source 7B models (like `LLAMA2` or `BLOOM`) on consumer-grade hardware, provided they employ the correct adaptation strategy.
*   **The Death of "One-Size-Fits-All" Prompting:** The failure of simple prompting to match fine-tuned performance (Section 5.2) implies that for high-stakes translation, treating LLMs as black boxes is insufficient. The field must move towards **weight-based specialization**, where the model's internal representations are explicitly reshaped for the target task.
*   **Redefining Evaluation Metrics:** The discovery of **off-target translation** rates as high as **98.3%** (Table 3) exposes a critical blind spot in current evaluation practices. Relying solely on average metrics like BLEU or COMET can mask catastrophic failures. Future research must incorporate **stability metrics** (e.g., language identification consistency) as a primary filter before assessing semantic quality.

### 7.2 Critical Follow-Up Research Directions
The limitations and discoveries in this paper open several urgent avenues for future investigation:

**1. Breaking the Error Propagation Loop without 4x Cost**
The study identifies **autoregressive error propagation** as the root cause of off-target failures and demonstrates that regenerating context (**REGEN**) fixes the issue but increases inference cost by **400%** (Table 4).
*   *Research Question:* Can we design **architectural interventions** or **decoding algorithms** that break this feedback loop without re-computing the entire context?
*   *Potential Directions:* Investigating **non-autoregressive decoding** for document context, introducing **explicit memory modules** that store verified context states, or developing **confidence-based gating mechanisms** that only trigger regeneration when the model detects a potential language drift.

**2. Understanding the "Instruction Tuning Penalty"**
The counter-intuitive finding that **base models** (`LLAMA2`) outperform **instruction-tuned models** (`VICUNA`) for DOCMT (Table 8) challenges the prevailing dogma that instruction tuning universally improves downstream performance.
*   *Research Question:* What specific features of instruction tuning (e.g., conversational bias, refusal patterns) conflict with the rigid structural requirements of document translation?
*   *Potential Directions:* Analyzing the latent space of instruction-tuned vs. base models to identify "translation-negative" vectors. This could lead to **"de-alignment" techniques** or hybrid training objectives that preserve multilingual flexibility while retaining instruction-following capabilities.

**3. Scaling Laws for Stability**
Since this study was constrained to **7B models** due to resources (Section 8), it remains unknown if the observed instability (non-convergence, off-target drift) persists at larger scales.
*   *Research Question:* Does scaling to 13B, 70B, or larger parameters naturally resolve the error propagation and training instability issues, or are these fundamental flaws in autoregressive document generation?
*   *Potential Directions:* Replicating this two-stage fine-tuning protocol on larger open-weight models (e.g., `LLAMA2-70B`, `Mixtral`) to determine if the "sweet spot" for stable DOCMT shifts with scale.

**4. Long-Context Discourse Modeling**
The current approach limits context to **three preceding sentences** (Section 3.1). This is insufficient for documents where coherence spans paragraphs or chapters (e.g., novels, legal contracts).
*   *Research Question:* How can LLMs be adapted to maintain consistency over **long-range dependencies** without overwhelming the context window or exacerbating error propagation?
*   *Potential Directions:* Exploring **hierarchical attention mechanisms** or **retrieval-augmented generation (RAG)** strategies where the model dynamically retrieves relevant past sentences from the document rather than relying on a fixed sliding window.

### 7.3 Practical Applications and Downstream Use Cases
The specific strengths identified in this work point to immediate practical applications where adapted 7B LLMs offer a distinct advantage over traditional systems:

*   **Low-Resource & Niche Domain Translation:**
    Due to the superior **out-of-domain generalization** observed on WMT2023 (Table 7), these models are ideal for translating content in domains where parallel data is scarce but the backbone LLM has broad world knowledge (e.g., translating technical manuals, niche scientific papers, or historical texts). Traditional encoder-decoder models (like MT5) tend to overfit to their training domain (TED talks), whereas adapted LLMs leverage their pre-training to handle unseen vocabulary and styles.

*   **Privacy-Sensitive On-Device Translation:**
    The success of **LoRA fine-tuning** (updating only ~8M parameters) means that high-quality DOCMT models can be deployed on edge devices or within secure, air-gapped environments. Organizations can fine-tune a base 7B model on their proprietary document corpus without sending sensitive data to external APIs, achieving better context awareness than sentence-level on-device models.

*   **Literary and Subtitle Translation:**
    The demonstrated ability to handle **discourse phenomena** (pronoun resolution, lexical cohesion) better than sentence-level models (Table 5) makes these adapted LLMs particularly suitable for **literary translation** and **subtitle generation**, where maintaining character voice and narrative flow across sentences is critical. The "Selective Excellence" in X-En tasks suggests they are ready for assisting human translators in translating foreign literature into English.

### 7.4 Reproducibility and Integration Guidance
For practitioners looking to integrate these findings, the paper provides a clear decision matrix based on data availability and resource constraints:

| Scenario | Recommended Approach | Rationale |
| :--- | :--- | :--- |
| **Abundant Parallel Data** (>20k docs) | **LoRA Fine-Tuning** on Base Model | LoRA prevents overfitting on large datasets and achieves higher peak performance (Figure 4). Base models (`LLAMA2`) show better transfer than instruction-tuned ones. |
| **Scarce Parallel Data** (<2k docs) | **Full Fine-Tuning (FFT)** on Base Model | FFT is vastly more data-efficient, reaching near-peak performance with only **1%** of the data, whereas LoRA requires **10%** (Figure 4). |
| **Multilingual Source Languages** | **BLOOM-7B Backbone** | `BLOOM`'s multilingual pre-training significantly reduces **off-target translation rates** compared to English-centric backbones like `LLAMA2` (Table 3). |
| **High-Stakes Production** | **Hybrid Decoding Strategy** | Do not rely solely on **REUSE** (standard autoregressive) due to error propagation risks. Implement a lightweight **language detection check** on generated context; if drift is detected, switch to **REGEN** (regenerate context) for that segment despite the higher cost. |
| **Prompt Design** | **Prompt 4 Structure** | Always use the compound structure: **Alternating Source-Target Pairs + Explicit Instruction**. Avoid block-text contexts or instructions alone, as the combination yields the best stability (Table 1). |

**Integration Warning:** When deploying these models, practitioners must implement a **guardrail for language identification**. As shown in Table 3, even well-tuned models can suffer sudden off-target collapse. A simple fastText language detector running on the output stream can flag instances where the model drifts, allowing the system to fallback to a specialized encoder-decoder model or trigger a regeneration step, ensuring reliability without sacrificing the LLM's superior discourse capabilities.

In conclusion, this work establishes that the path to robust, document-level translation with LLMs is not through bigger models, but through **smarter adaptation**. By addressing the specific failure modes of error propagation and leveraging the data efficiency of full fine-tuning in low-resource settings, the field can unlock the potential of moderately-sized models to deliver translation quality that rivals the largest proprietary systems.