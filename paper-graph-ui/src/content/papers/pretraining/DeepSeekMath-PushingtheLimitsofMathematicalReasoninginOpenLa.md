# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

**ArXiv:** [2402.03300](https://arxiv.org/abs/2402.03300)

## 🎯 Pitch

DeepSeekMath introduces the first open 7B-parameter language model that surpasses 50% Top-1 accuracy on the challenging MATH benchmark, achieved by meticulously curating a massive 120B-token math-focused web dataset and pioneering the memory-efficient Group Relative Policy Optimization (GRPO) algorithm for reinforcement learning. This innovation narrows the gap with proprietary giants like GPT-4, offering the research community a scalable foundation for advanced mathematical reasoning and robust multi-step problem solving—critical for science, education, and AI alignment.

---

## 1. Executive Summary

This paper introduces **DeepSeekMath 7B**, a domain-specific language model that continues pre-training DeepSeek-Coder-Base-v1.5 7B on 120B math-related tokens sourced from Common Crawl through an iterative fastText-based data selection pipeline, combined with natural language and code data. DeepSeekMath-Base achieves 64.2% on GSM8K and 36.2% on the competition-level MATH benchmark using chain-of-thought prompting, outperforming the 540B-parameter Minerva model and establishing that a smaller model pre-trained on high-quality data can rival or exceed models ~77× larger. The paper further introduces **Group Relative Policy Optimization (GRPO)** — a variant of PPO that foregoes the critic model by estimating the baseline from group scores (normalizing rewards within a group of sampled outputs for the same question) — which boosts DeepSeekMath-Instruct from 46.8% to 51.7% on MATH using only in-domain instruction-tuning data, marking the first time an open-source model exceeds 50% on this benchmark. Analysis reveals that GRPO enhances Maj@K but not Pass@K, establishing that the RL gains arise from making the output distribution more robust rather than expanding the model's fundamental reasoning capacity.

## 2. Context and Motivation

### The Core Problem: Open-Source Models Lag Significantly in Mathematical Reasoning

The central problem this paper addresses is straightforward to state but difficult to solve: **open-source language models dramatically underperform proprietary models on mathematical reasoning benchmarks**, and the community lacks a clear, replicable recipe for closing this gap. At the time of writing (early 2024), closed-source models like GPT-4 and Gemini-Ultra achieve MATH benchmark scores exceeding 50%, while the best open-source models — even those substantially larger — remain stuck below roughly 35% (Table 5). This represents more than just a performance gap; it is a capability gap with downstream consequences for research reproducibility, academic access, and the development of mathematical AI assistants that can be deployed without reliance on proprietary APIs.

The paper frames this in explicitly practical terms in Section 1:

> "cutting-edge models such as GPT-4 and Gemini-Ultra are not publicly available, and the currently accessible open-source models considerably trail behind in performance."

This gap matters for several reasons that extend beyond benchmark bragging rights:

- **Scientific reproducibility**: Closed models cannot be inspected, modified, or studied by the broader research community. Every paper that builds on GPT-4's mathematical reasoning capabilities inherits an opaque foundation — the model's training data composition, architecture decisions, and failure modes are all inaccessible.
- **Domain-specific deployment**: Mathematical reasoning is not just a research curiosity. It underpins educational technology (automated tutoring), scientific computing (formal verification, theorem proving), and quantitative analysis in finance and engineering. These applications often require models that can run on consumer hardware or within institutional compute budgets, making the 7B parameter scale particularly attractive.
- **The self-improvement bottleneck**: A recurring vision in the LLM literature is that models can generate their own training data for iterative improvement (as explored in works like STaR and ReST$^{EM}$). But this vision depends on the model already being competent enough at a task to generate correct solutions at a reasonable rate — a condition that was not met by any open-source model on the MATH benchmark prior to DeepSeekMath.

The paper's contribution is therefore both empirical (they actually close a substantial portion of the gap) and methodological (they provide a detailed, reproducible recipe that combines data engineering with a novel reinforcement learning approach).

### Why Existing Approaches Fall Short

The paper's motivation is built on diagnosing specific weaknesses in prior approaches to mathematical language modeling. These weaknesses fall into three categories:

**1. Inadequate Pre-Training Data Strategy for Mathematics**

Prior to DeepSeekMath, the dominant approaches to building math-capable LMs used pre-training corpora that were either too small, of insufficient quality, or both. The paper explicitly benchmarks against three representative corpora in Section 2.2:

- **MathPile** (Wang et al., 2023c): 8.9B tokens, with over 85% sourced from arXiv. Table 1 shows that training on MathPile produces *worse* performance than no math training at all on several benchmarks (GSM8K drops from 2.9% to 2.7%; CMATH drops from 12.3% to 1.2%). This is a striking negative result that the paper returns to in Section 5.1.2 — arXiv papers, despite being rich in mathematical notation, appear largely ineffective at improving downstream mathematical reasoning. The paper hypothesizes that arXiv content may be too far removed from the problem-solving format tested by benchmarks, but this finding alone is a significant practical insight for anyone building math corpora.

- **OpenWebMath** (Paster et al., 2023): 13.6B tokens from Common Crawl filtered for mathematical content. This corpus is substantially better than MathPile (GSM8K: 11.5% vs. 2.7%, MATH: 8.9% vs. 3.3% in Table 1), demonstrating that web-scraped math pages contain useful signal. However, its limited size means that training curves plateau quickly — Figure 3 shows that after roughly 50B tokens of training on OpenWebMath (i.e., multiple epochs of the same data), performance gains flatten.

- **Proof-Pile-2** (Azerbayev et al., 2023): 51.9B tokens combining OpenWebMath, AlgebraicStack (10.3B tokens of mathematical code), and arXiv papers. This corpus was used to train Llemma 7B and 34B, which were the state-of-the-art open-source math models prior to DeepSeekMath. Table 1 shows Proof-Pile-2 outperforms the smaller corpora (GSM8K: 14.3%, MATH: 11.2%), but Figure 3 demonstrates that its performance also plateaus as the training budget increases.

The critical insight from Figure 3 is that DeepSeekMath Corpus, at 120.2B tokens, shows a **steeper learning curve and more sustained improvement** compared to all baselines. At 50B tokens of training (one full epoch of Proof-Pile-2), the DeepSeekMath Corpus-trained model already outperforms the Proof-Pile-2-trained model across all benchmarks, indicating a genuine quality advantage rather than mere scale. The paper attributes this to the iterative data selection pipeline that progressively improves the fastText classifier by incorporating human-annotated domain knowledge, effectively doing a more thorough job of separating high-quality mathematical content from superficially math-like but low-quality pages.

**2. The Unresolved Question of Code Training and Reasoning**

A long-standing but unverified hypothesis in the LLM community holds that training on code improves general reasoning capabilities. The paper explicitly names this as a motivating question in Section 5.1.1:

> "A popular yet unverified hypothesis suggests that code training improves reasoning. We attempt to offer a partial response to this, particularly within the mathematical domain: code training improves models' ability to do mathematical reasoning both with and without tool use."

The paper tests this by initializing DeepSeekMath-Base from DeepSeek-Coder-Base-v1.5 7B (a model pre-trained on code) rather than from a general-domain base model. The comparison in Table 6 shows that a two-stage training pipeline (code training → math training) substantially outperforms a general training → math training pipeline, and also outperforms math-only training. This finding has practical implications for model architecture decisions: it suggests that organizations building math-focused models should start from a code-capable base, not a general language model.

**3. The Limitations of Instruction Tuning Alone for Mathematical Reasoning**

Even after pre-training a strong base model, instruction tuning typically produces performance that falls well short of what is possible. Table 5 shows that DeepSeekMath-Instruct achieves 46.8% on MATH with chain-of-thought reasoning — excellent by open-source standards, but still 5–6 percentage points behind GPT-4 and Gemini Ultra.

The paper identifies that prior RL-based approaches to math improvement (e.g., WizardMath using PPO, Math-Shepherd-Mistral using process-supervised PPO) suffer from a specific practical problem: **PPO requires a value function model of comparable size to the policy model**, roughly doubling the memory footprint and computational cost of training. For the DeepSeekMath team working with limited resources, this overhead represents a genuine barrier. The paper describes GRPO specifically as a response to this constraint:

> "GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources."

This is not merely an engineering convenience. By eliminating the value model, GRPO also sidesteps a known difficulty with PPO in the LLM setting: the value function must predict per-token advantages even though explicit rewards typically only arrive at the final token (Equation 2). Training a value function that accurately decomposes a single end-of-sequence reward into per-step credit assignments is challenging, and GRPO's group-based relative scoring provides an alternative mechanism that may be more natural for comparative reward models.

### The Deeper Gap: Understanding *Why* RL Works for Math

Beyond the practical gap of model performance, the paper addresses a conceptual gap in the literature: **there is no unified framework for understanding why different post-training methods (SFT, RFT, DPO, PPO, GRPO) work, and under what conditions each is most effective**. Section 5.2.1 presents this as a motivating observation:

> "there exist three key components: 1) Data Source, which determines the training data; 2) Reward Function, which is the source of the training reward signal; 3) Algorithm: which processes the training data and the reward signal to the gradient coefficient that determines the magnitude of the penalty or reinforcement for the data."

The paper shows that methods previously treated as conceptually distinct (RFT, DPO, PPO) can all be understood as variants along these three axes. For example:

- **RFT vs. Online RFT**: Both use rule-based reward (answer correctness), but RFT uses stale samples from the initial SFT model while Online RFT uses fresh samples from the current policy model. Figure 5 shows Online RFT substantially outperforms RFT, indicating that the data source axis (online vs. offline sampling) is critical.

- **Online RFT vs. GRPO**: Both use online samples, but Online RFT's gradient coefficient is binary (1 for correct, 0 for incorrect, per Equation 10), while GRPO's gradient coefficient is continuous and can be negative (Equation 21). This means GRPO can penalize incorrect responses proportionally to how wrong they are, and can differentially reinforce correct responses based on their quality. Figure 5 shows GRPO outperforms Online RFT, indicating that the algorithm axis (how the reward signal is transformed into gradient updates) matters.

This unified paradigm is not just a theoretical nicety — it provides a diagnostic toolkit for understanding *why* RL works. The paper's key finding in Section 5.2.2 and Figure 7 is that GRPO improves Maj@K (majority voting over K samples) but not Pass@K (whether at least one of K samples is correct). This means RL is not expanding the model's capability frontier (it doesn't become capable of solving fundamentally new types of problems) but rather is **redistributing probability mass within the already-achievable solution space to make correct answers more likely to appear in the top-K samples**. The paper connects this to the "misalignment problem" identified by Wang et al. (2023a): SFT models often have the raw knowledge to solve problems (as evidenced by non-zero Pass@K) but fail to consistently output correct solutions because their output distribution misranks reasoning paths.

### How This Paper Positions Itself

The paper positions itself at the intersection of two research thrusts:

**1. Scalable math pre-training.** It extends the line of work from Minerva (Lewkowycz et al., 2022a), Llemma (Azerbayev et al., 2023), and OpenWebMath (Paster et al., 2023) by demonstrating that the key bottleneck is not model scale but data quality and data scale on the web. The paper's data collection pipeline (Figure 2) is presented as a generalizable method for domain-specific corpus construction that could be applied to coding, science, or any other domain where high-quality web content exists but is interleaved with noise at scale. The iterative fastText approach — train a classifier, recall documents, identify high-density domains, manually annotate URL patterns, enrich the seed corpus, retrain the classifier — is a concrete engineering contribution that substantially advances over the one-shot filtering used in prior work.

**2. RL for reasoning improvement.** It extends the line of work from WizardMath (Luo et al., 2023) and Math-Shepherd (Wang et al., 2023b), but with two distinctive contributions: the GRPO algorithm itself (a practical innovation that reduces memory cost), and the unified analytical framework (a conceptual innovation that explains *why* different RL variants behave differently). The paper explicitly frames GRPO as an *efficient* alternative to PPO, not necessarily a *better* one in terms of final performance — the motivation is resource reduction, supported by the observation that GRPO achieves strong gains using only a subset of the instruction-tuning data.

The paper also positions itself as a **reconciliation of conflicting findings about arXiv data**. Section 5.1.2 acknowledges that arXiv papers are included in nearly every major math pre-training corpus (Minerva, Llemma, MathPile) despite limited evidence of their effectiveness. The paper's experiments across two model scales and two arXiv corpora consistently show "no notable improvements or even deterioration" (Table 8), while carefully noting the limitations of this conclusion (Section 5.1.2: arXiv might help on tasks not evaluated here, at larger model scales, or when combined with other data in specific ratios). This finding is positioned as counterintuitive and worthy of further investigation.

Finally, the paper positions its RL analysis (Section 5.2.3) as forward-looking: it identifies three specific research directions (data source expansion beyond in-distribution prompts, robust algorithms that can handle noisy rewards, and better reward model generalization) that it argues are necessary for the next generation of RL-enhanced language models. The weakness-to-strong alignment framing (Burns et al., 2023) is explicitly invoked, connecting the paper's findings to a broader research agenda in AI alignment and capability amplification.

## 3. Technical Approach

### 3.1 Reader Orientation

DeepSeekMath is a **7-billion-parameter language model specialized for mathematical reasoning**, built by continuing to train an existing code-focused model on a massive, carefully curated dataset of 120 billion math-related tokens extracted from the public web, then further refined through supervised instruction tuning and a novel reinforcement learning algorithm called Group Relative Policy Optimization (GRPO). The system solves the problem of open-source models dramatically underperforming proprietary ones on math benchmarks: rather than scaling model size (the approach of prior work like Minerva 540B), DeepSeekMath scales data quality and data quantity on the web through an iterative, human-in-the-loop filtering pipeline, and then uses GRPO—a more memory-efficient variant of PPO that eliminates the critic model by normalizing rewards within groups of sampled answers—to redistribute the model's probability mass toward correct solutions without requiring a separate value function.

### 3.2 Big-Picture Architecture (Diagram in Words)

The DeepSeekMath system has five major stages, arranged sequentially:

1. **Iterative Data Collection Pipeline (Section 2.1):** A fastText classifier, bootstrapped from a seed corpus (OpenWebMath) and progressively refined through human annotation of URL patterns, mines 120B tokens of mathematical web pages from 40B deduplicated Common Crawl HTML documents over four iterations. This produces the DeepSeekMath Corpus.

2. **Continued Pre-Training (Section 2.3):** Starting from DeepSeek-Coder-Base-v1.5 7B (a code-pretrained model), the system trains for 500B additional tokens on a mixture of 56% DeepSeekMath Corpus, 4% AlgebraicStack, 10% arXiv, 20% GitHub code, and 10% natural language Common Crawl data. This produces DeepSeekMath-Base 7B.

3. **Supervised Fine-Tuning (Section 3):** DeepSeekMath-Base is instruction-tuned on 776K examples covering chain-of-thought, program-of-thought, and tool-integrated reasoning formats in English and Chinese. This produces DeepSeekMath-Instruct 7B.

4. **Reinforcement Learning with GRPO (Section 4):** Using only ~144K chain-of-thought questions from GSM8K and MATH (a subset of the SFT data), the system applies Group Relative Policy Optimization—a variant of PPO that replaces the learned value function with a group-based baseline computed by normalizing rewards across multiple sampled outputs for each question. This produces DeepSeekMath-RL 7B.

5. **Evaluation (Sections 2.3, 3.2, 4.2):** The models are evaluated on English and Chinese math benchmarks (GSM8K, MATH, SAT, OCW Courses, MMLU-STEM, CMATH, Gaokao-MathCloze, Gaokao-MathQA) under chain-of-thought, program-of-thought, and tool-integrated reasoning settings, plus natural language understanding (MMLU, BBH) and code (HumanEval, MBPP).

Information flows linearly: Common Crawl → DeepSeekMath Corpus → DeepSeekMath-Base → DeepSeekMath-Instruct → DeepSeekMath-RL → Benchmark Evaluations. The key meta-insight is that each stage builds on the previous one, and the final RL stage provides gains *without* expanding the model's fundamental capability frontier—it redistributes probability mass within the already-attainable solution space.

### 3.3 Roadmap for the Deep Dive

The technical breakdown follows the chronological construction of the system, which also reflects the logical dependencies:

- **First, the data collection pipeline (Section 2.1):** This is the foundational innovation—without high-quality math data at scale, none of the downstream results are possible. We trace the four-iteration fastText loop in detail because understanding *why* the data is high quality requires understanding the iterative enrichment mechanism.

- **Second, pre-training and its key design choices (Sections 2.2–2.3):** With the corpus in hand, we examine how the model is trained, why starting from a code model matters, and what ablation studies reveal about arXiv papers and code-math interaction.

- **Third, supervised fine-tuning (Section 3):** A relatively standard stage, but we cover the data composition and training configuration because it establishes the baseline that RL must improve upon.

- **Fourth, Group Relative Policy Optimization (Section 4.1):** The most algorithmically novel contribution. We walk from PPO's objective, through the motivation for removing the critic, to the GRPO objective and its variants (outcome supervision, process supervision, iterative RL). Every equation receives the full treatment.

- **Fifth, the unified RL paradigm (Section 5.2.1 and Appendix A.1):** Not a system component per se, but an analytical framework that explains *why* GRPO works relative to alternatives. This draws on the gradient coefficient formulation in Equation 5.

- **Sixth, the empirical investigation of why RL helps (Section 5.2.2):** The Pass@K vs. Maj@K distinction resolves the apparent puzzle of RL improving benchmark scores without expanding fundamental capability.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **empirical systems paper** whose core contributions are: (1) a reproducible, iterative pipeline for extracting high-quality mathematical training data from Common Crawl at 120B-token scale, (2) evidence that training a smaller model on higher-quality data can match or exceed models ~77× larger, (3) a memory-efficient RL algorithm (GRPO) that achieves significant gains on math benchmarks using only in-domain data, and (4) a unified analytical framework explaining why different post-training methods work.

---

#### The Iterative Data Collection Pipeline (DeepSeekMath Corpus Construction)

The data collection pipeline, depicted in Figure 2, solves a specific problem: **Common Crawl contains valuable mathematical web pages, but they are interleaved with massive amounts of non-mathematical content and low-quality pseudo-math pages.** A single-pass classifier, no matter how well-trained, will either miss substantial mathematical content (low recall) or include too much noise (low precision). The iterative approach progressively improves both precision and recall by using human feedback to expand the seed corpus.

**Step 1: Initial seed corpus selection.** The pipeline begins with OpenWebMath (Paster et al., 2023), a collection of 13.6B tokens of high-quality mathematical web text that was itself extracted from Common Crawl using a combination of heuristic filtering and classifier-based selection. OpenWebMath serves as the initial "positive example" set—it defines what the pipeline considers desirable mathematical content. While OpenWebMath is already filtered, it represents only a fraction of the mathematical content available on the web; the goal is to use it as a launching point to discover much more.

**Step 2: Initial fastText model training.** A fastText classifier (Joulin et al., 2016) is trained to distinguish mathematical web pages from non-mathematical ones. The training data consists of 500,000 randomly selected pages from OpenWebMath as positive examples and 500,000 randomly selected pages from Common Crawl as negative examples. The fastText model is configured with:

- Vector dimension: 256
- Learning rate: 0.1
- Maximum length of word n-gram: 3
- Minimum number of word occurrences: 3
- Number of training epochs: 3

The authors use an open-source library for training. The fastText architecture is chosen because it is computationally efficient at scale—it can rapidly score billions of documents—and its n-gram features capture character-level patterns that are highly informative for distinguishing mathematical text (which contains LaTeX notation, mathematical symbols, and distinctive formatting) from general web content.

**Step 3: Document scoring and ranking.** Before classification, the raw Common Crawl is deduplicated using URL-based exact deduplication and near-deduplication techniques, reducing it to approximately 40 billion HTML pages. The fastText model then scores every page, producing a scalar confidence value for how "math-like" each page is. Pages are ranked by this score, and the top-ranking pages are preserved. The volume preserved is determined by pre-training experiments on the top 40B, 80B, 120B, and 160B tokens. In the first iteration, the top 40B tokens are kept.

**Step 4: Domain-level enrichment (the key to iterative improvement).** After the first iteration, many mathematical pages remain uncollected because the fastText model, trained only on OpenWebMath, lacks diversity in its positive examples—it learns to recognize pages that look like OpenWebMath specifically, not mathematical pages in general. To expand the seed corpus, the pipeline performs a domain-level analysis:

- The entire Common Crawl is organized into disjoint **domains**, where a domain is defined as all web pages sharing the same base URL (e.g., `mathoverflow.net`, `math.stackexchange.com`).
- For each domain, the pipeline computes the **percentage of pages that were collected in the first iteration** (i.e., were ranked in the top 40B tokens by the fastText model).
- Domains where over 10% of pages were collected are classified as **math-related domains**. This threshold is a design choice: if a substantial fraction of a domain's pages score highly under the classifier, the domain likely contains genuine mathematical content, and the uncollected pages in that domain are likely false negatives (mathematical pages the classifier missed).
- Human annotators then manually label specific **URL paths** within these identified domains that correspond to mathematical content (e.g., `mathoverflow.net/questions` rather than `mathoverflow.net/help`). This is a critical human-in-the-loop step: it injects domain knowledge about which *parts* of a generally mathematical website contain actual mathematical content.
- Web pages linked to these annotated URL paths that were not collected in the first iteration are added to the seed corpus as new positive examples.

**Step 5: Classifier retraining and iteration.** The enriched seed corpus (original OpenWebMath + newly discovered pages from math-related domains) is used to train an improved fastText model. This model has broader coverage of mathematical content because its positive examples now span a more diverse set of mathematical websites. The new classifier is used to recall additional mathematical pages from Common Crawl, and the process repeats. After four iterations of this cycle (classify → rank → identify domains → annotate URLs → enrich seed → retrain), the pipeline converges: in the fourth iteration, nearly 98% of the data was already collected in the third iteration, indicating that the classifier has reached near-exhaustive recall of the targeted mathematical web content. The final corpus contains 35.5 million mathematical web pages totaling 120 billion tokens.

**Why this iterative approach works better than one-shot classification.** A one-shot classifier trained only on OpenWebMath would learn a narrow definition of "mathematical content" biased toward the specific websites, formatting styles, and mathematical subfields present in OpenWebMath. The iterative enrichment progressively expands this definition: each iteration discovers new mathematical domains, the human annotators identify the relevant URL patterns within those domains, and the retrained classifier learns to recognize a more diverse set of mathematical pages. This can be understood as a form of **active learning at the domain level**—the system identifies regions of the web where the current classifier is likely making false-negative errors (domains with >10% collected pages but many uncollected ones), solicits human feedback to correct those errors (URL annotation), and updates the classifier accordingly.

**Decontamination.** To avoid benchmark contamination, the pipeline filters out any web page containing questions or answers from evaluation benchmarks. The filtering criteria are:

- Any text segment containing a 10-gram string that matches exactly with any substring from GSM8K, MATH, CMATH, or AGIEval is removed.
- For benchmark texts shorter than 10 grams but at least 3 grams, exact matching is used to filter out contaminated pages.

This is a conservative decontamination approach: it errs on the side of removing potentially useful data to ensure benchmark integrity. A 10-gram overlap is a very strict criterion—a 10-word sequence from a MATH problem appearing in a web page is almost certainly direct copying rather than coincidental overlap. The 3-gram threshold for shorter texts handles edge cases like very short problem statements.

**Scale and multilingualism.** The final DeepSeekMath Corpus (120.2B tokens) is approximately 7× larger than the math web pages used by Minerva and 9× larger than OpenWebMath. Unlike Minerva and Llemma, which focused exclusively on English mathematical content, the DeepSeekMath Corpus naturally includes multiple languages because Common Crawl contains web pages in many languages, and the iterative pipeline does not apply language filtering. The paper notes (Section 2.2.2) that English and Chinese are "the two most represented languages," which proves important for downstream Chinese math benchmark performance (Table 1 shows CMATH improving from 12.3% with no math training to 41.5% with DeepSeekMath Corpus training, while English-only corpora like Proof-Pile-2 only reach 19.9%).

---

#### Pre-Training Data Mix and Configuration

**Base model initialization.** DeepSeekMath-Base 7B is initialized from DeepSeek-Coder-Base-v1.5 7B (Guo et al., 2024), not from a general-purpose language model. This is a deliberate design choice motivated by the hypothesis that code training benefits mathematical reasoning. The paper tests this hypothesis explicitly in Section 5.1.1 (Table 6, Table 7) and finds that a two-stage training pipeline (code training → math training) outperforms both general training → math training and math-only training. The intuition is that code pre-training teaches the model structural reasoning patterns (variable manipulation, symbolic logic, step-by-step algorithmic thinking) that transfer to mathematical proof construction and equation manipulation.

**Training data distribution.** The model is trained for 500B tokens with the following data mixture:

- 56% from the DeepSeekMath Corpus (mathematical web pages)
- 4% from AlgebraicStack (mathematical code in languages like Lean, Isabelle, and Python)
- 10% from arXiv (scientific papers, predominantly LaTeX-formatted mathematics and physics)
- 20% from GitHub code (general programming code across multiple languages)
- 10% natural language data from Common Crawl in English and Chinese

The inclusion of code (20% GitHub + 4% AlgebraicStack) is designed to maintain and enhance the code reasoning capabilities inherited from DeepSeek-Coder-Base-v1.5, since math-only training can cause catastrophic forgetting of code skills (as shown in Table 7, where math training after code training drops HumanEval from 25.0% to 12.2%). The 10% natural language Common Crawl data serves as a regularizer to maintain general language understanding. The arXiv data is included despite the paper's finding (Section 5.1.2) that arXiv alone does not improve math benchmarks, because the ablation studies were conducted at smaller scales and the paper leaves open the possibility that arXiv tokens might be beneficial when combined with other data types or at larger model scales.

**Training hyperparameters.** The training configuration follows the DeepSeek LLM training recipe (DeepSeek-AI, 2024):

- Optimizer: AdamW (Loshchilov and Hutter, 2017) with β₁ = 0.9, β₂ = 0.95, weight_decay = 0.1
- Learning rate schedule: multi-step, where the learning rate reaches its peak after 2,000 warmup steps, decreases to 31.6% of the peak after 80% of training, and further decreases to 10.0% of the peak after 90% of training
- Maximum learning rate: 4.2e-4 (lower than the 5.3e-4 used for the 1.3B ablation experiments, since larger models typically require lower peak learning rates for stability)
- Batch size: 10M tokens (substantially larger than the 4M tokens used for the 1.3B experiments, since larger models benefit from larger batches for gradient stability)
- Context length: 4K tokens
- Training framework: HAI-LLM (High-flyer, 2023), described as "efficient and light-weight"

The multi-step learning rate schedule is a practical choice: it maintains a high learning rate through most of training for rapid progress, then decays aggressively toward the end to settle into a good local minimum. The specific decay factors (31.6% and 10.0%) are standard in the DeepSeek LLM training recipe.

**Validation experiments at 1.3B scale.** Before committing 500B tokens of 7B-parameter training, the paper validates the corpus quality using a smaller 1.3B-parameter model (DeepSeek-LLM 1.3B) trained for 150B tokens on each candidate corpus. This is a cost-effective proxy: if the DeepSeekMath Corpus shows advantages at 1.3B scale, those advantages are likely to persist (and potentially compound) at 7B scale. Table 1 and Figure 3 present these results, confirming that the DeepSeekMath Corpus yields steeper learning curves and higher final performance than MathPile, OpenWebMath, or Proof-Pile-2 at matched training budgets.

---

#### Code Training and Mathematical Reasoning (Ablation Design)

The paper investigates whether code training benefits mathematical reasoning through controlled experiments at the 1.3B scale. The experimental design in Section 5.1.1 compares four training configurations:

**Two-stage training (Code → Math):** DeepSeek-LLM 1.3B is trained on 400B code tokens, then an additional 150B math tokens. The 400B code pre-training is substantial—nearly 3× the subsequent math training—to ensure the code capabilities are well-developed before math specialization begins.

**Two-stage training (General → Math):** As a control, DeepSeek-LLM 1.3B is trained on 400B general-domain tokens (sampled from DeepSeek-AI's large-scale general corpus), then 150B math tokens. This isolates the effect of *code specifically* versus *any pre-training data* on subsequent math training.

**One-stage math-only training:** DeepSeek-LLM 1.3B is trained on 150B math tokens directly, with no prior code or general-domain training. This is the baseline for "what happens if you just do math training from scratch."

**One-stage mixed training (Code + Math):** DeepSeek-LLM 1.3B is trained on a mixture of 400B code tokens and 150B math tokens in a single stage (total 550B tokens). This tests whether interleaving code and math avoids catastrophic forgetting of code capabilities while still benefiting math reasoning.

**Results (Table 6 and Table 7).** The key findings are:

- **Code → Math two-stage training achieves the best math-without-tool performance** (GSM8K: 21.9%, MATH: 15.3%, CMATH: 39.7%), outperforming General → Math (19.1%, 14.4%, 37.2%) and math-only (20.5%, 13.1%, 37.6%). The advantage is modest but consistent across benchmarks.

- **Code → Math two-stage training achieves the best math-with-tool performance** (GSM8K+Python: 17.4%, MATH+Python: 9.4%), substantially outperforming General → Math (14.3%, 6.7%) and math-only (11.4%, 6.5%). The advantage here is larger, which makes intuitive sense: code pre-training directly teaches the model to write correct Python, which is then leveraged when math problems are solved via Python programs.

- **Code → Math two-stage training causes catastrophic forgetting of code capabilities.** Table 7 shows that HumanEval drops from 25.0% after code training to 12.2% after subsequent math training, and MBPP drops from 40.0% to 17.0%. This is expected: the math training phase does not include code data, so the model's code generation skills degrade.

- **One-stage mixed training preserves code capabilities while achieving good math performance.** Table 7 shows HumanEval at 29.3% (better than code-only!) and MBPP at 39.4% (nearly matching code-only). Table 6 shows math-with-tool performance (GSM8K+Python: 19.7%, MATH+Python: 13.5%) actually exceeding the two-stage approach, though math-without-tool performance (GSM8K: 17.6%, MATH: 12.1%) is somewhat lower.

- **One-stage mixed training compromises math-without-tool performance.** The paper conjectures that "DeepSeek-LLM 1.3B, due to its limited scale, lacks the capacity to fully assimilate both code and mathematical data simultaneously." This is a capacity-limitation hypothesis: at 1.3B parameters, the model cannot learn both code and math patterns at full fidelity simultaneously, so mixing them forces a tradeoff. At 7B scale (which DeepSeekMath-Base uses), this tradeoff may be less severe, which is why the final 7B model uses a mixed-data approach (20% code, 56% math, etc.) rather than pure two-stage training.

**Why code training helps math.** The paper does not provide mechanistic evidence for the transfer, but the results support the hypothesis that code training teaches structured reasoning patterns that transfer to mathematical problem-solving. Code requires precise syntax, variable scoping, step-by-step execution, and logical conditionals—all skills that are also essential for constructing mathematical proofs and solving multi-step quantitative problems. The fact that the benefit is larger for tool-use (Python-assisted) math makes the transfer even clearer: code training directly teaches the programming skills needed to write Python solutions to math problems.

---

#### arXiv Paper Ablation (Negative Result)

The paper conducts a targeted ablation to test whether arXiv papers—which are included in nearly every major math pre-training corpus (Minerva, Llemma, MathPile)—actually improve mathematical reasoning. The results are consistently negative.

**Corpora tested.** Two arXiv-derived corpora are evaluated:

- **MathPile** (Wang et al., 2023c): 8.9B tokens total, over 85% from arXiv, with the remainder from textbooks, Wikipedia, ProofWiki, CommonCrawl, and StackExchange. This corpus includes cleaning and filtering heuristics.

- **ArXiv-RedPajama** (Computer, 2023): 28.0B tokens, consisting of the entirety of arXiv LaTeX files with preambles, comments, macros, and bibliographies removed. This is a "raw" arXiv corpus with minimal processing beyond LaTeX cleanup.

**Experimental setup.** Two model scales are tested:

- DeepSeek-LLM 1.3B trained for 150B tokens on each arXiv corpus
- DeepSeek-Coder-Base-v1.5 7B trained for 40B tokens on each arXiv corpus

The different token counts (150B vs. 40B) reflect the different scales—the 1.3B model needs more tokens to reach convergence, while the 7B model (which is already pre-trained) needs fewer additional tokens.

**Results (Tables 8 and 9).** Across both model scales and both arXiv corpora, the results are remarkably consistent in their lack of improvement:

- On DeepSeek-LLM 1.3B: MathPile training slightly *reduces* GSM8K (2.9% → 2.7%) and CMATH (12.3% → 1.2%) compared to no math training. ArXiv-RedPajama shows minimal improvement on GSM8K (2.9% → 3.3%) and slight degradation on MMLU-STEM (19.5% → 9.0%) and CMATH (12.3% → 7.4%).

- On DeepSeek-Coder-Base-v1.5 7B: MathPile training reduces GSM8K (29.0% → 23.6%) and CMATH (45.9% → 37.9%). ArXiv-RedPajama shows similar performance to no math training on GSM8K (29.0% → 28.1%) but degrades CMATH (45.9% → 42.6%).

- On formal theorem proving (Table 9, miniF2F): MathPile reduces valid accuracy from 20.1% to 16.8% and test accuracy from 21.7% to 16.4%. ArXiv-RedPajama causes even larger drops (valid: 14.8%, test: 11.9%).

The paper's characterization is measured: "it seems that arXiv papers are ineffective in improving mathematical reasoning. When trained on an arXiv-only corpus, both models display no notable improvements or even deterioration across various mathematical benchmarks."

**Why might arXiv not help?** The paper does not provide a definitive explanation but offers contextual clues. The key distinction is between *containing mathematical notation* and *being useful for learning mathematical problem-solving*. arXiv papers are written in an expository, proof-theoretic style aimed at expert readers—they explain completed mathematical results, not the process of solving novel problems from scratch. In contrast, web pages on math forums (StackExchange, MathOverflow) and educational sites contain exactly the kind of step-by-step problem-solving reasoning that few-shot chain-of-thought prompting evaluates. The data distribution mismatch between arXiv's content format (theorems, proofs, definitions) and the benchmark format (solve this specific problem, show your work) may explain why training on arXiv alone doesn't transfer.

**Caveats the paper explicitly notes:**

- The impact of arXiv tokens on tasks not evaluated (e.g., theorem informalisation—converting formal proofs to natural language) is unknown.
- arXiv tokens might be beneficial when combined with other data types in specific ratios (the paper only tests arXiv-only training).
- The benefits of arXiv papers might manifest at larger model scales (beyond 7B).

The final DeepSeekMath-Base 7B training mixture includes 10% arXiv data despite this negative result, suggesting the authors believe arXiv might have value as a *component* of a diverse mixture even if it doesn't help in isolation. However, the result is important enough that the paper flags it prominently: researchers constructing math pre-training corpora should not assume arXiv inclusion is beneficial without empirical verification on their specific benchmarks.

---

#### Supervised Fine-Tuning (SFT) Data and Training

The SFT stage converts the base model (DeepSeekMath-Base 7B), which is trained only on next-token prediction, into an instruction-following model (DeepSeekMath-Instruct 7B) that can produce formatted solutions to mathematical problems.

**SFT data composition (776K total examples).** The instruction-tuning dataset is constructed from multiple sources covering English and Chinese mathematics:

- **English mathematical datasets:**
  - GSM8K and MATH problems annotated with **tool-integrated solutions** (where the model learns to interleave natural language reasoning with Python code execution). The annotation process is not described in detail, but the resulting format follows the approach of ToRA (Gou et al., 2023).
  - A subset of **MathInstruct** (Yue et al., 2023), a large-scale instruction-tuning dataset for mathematics containing problems solved with chain-of-thought (CoT) and program-of-thought (PoT) formats. MathInstruct aggregates data from multiple existing math datasets.
  - The training set of **Lila-OOD** (Mishra et al., 2022), a benchmark of out-of-distribution mathematical reasoning problems, with solutions in CoT or PoT format.

The English collection covers diverse mathematical fields including algebra, probability, number theory, calculus, and geometry.

- **Chinese mathematical datasets:**
  - Chinese K-12 mathematical problems (primary and secondary school level) spanning 76 sub-topics such as linear equations. Solutions are annotated in both CoT format and tool-integrated reasoning format. The collection methodology is not detailed, but the 76-topic taxonomy suggests a systematic effort to cover the Chinese mathematics curriculum.

The three solution formats taught are:

- **Chain-of-thought (CoT):** Step-by-step natural language reasoning, culminating in a final answer. This is the format evaluated in the "Chain-of-Thought Reasoning" rows of Table 5.

- **Program-of-thought (PoT):** The model writes a Python program that, when executed, produces the answer. Libraries like `math` and `sympy` can be used for complex computations. The execution result is evaluated as the answer. This is the format used in the "Tool-Integrated Reasoning" rows of Table 5.

- **Tool-integrated reasoning:** A hybrid format where natural language reasoning is interleaved with code execution, allowing the model to perform computations that would be impractical to do in its "head" while still explaining its reasoning process.

**SFT training configuration.**

- Training examples are randomly concatenated until reaching a maximum context length of 4K tokens. This is standard practice for instruction-tuning: rather than padding each example to the maximum length (which wastes compute), examples are packed sequentially into the full context window.
- Number of training steps: 500
- Batch size: 256 (effective batch size of 256 × 4K = 1M tokens per step, or 500M tokens total over the 500 steps)
- Learning rate: constant 5e-5 (no learning rate schedule—a common choice for SFT where the number of steps is relatively small)

The constant learning rate of 5e-5 is typical for fine-tuning: it is low enough to avoid catastrophic forgetting of pre-training knowledge but high enough to make meaningful progress in 500 steps. The total of 500M tokens of fine-tuning data is modest compared to the 500B tokens of pre-training—SFT is about teaching the model a specific output format and problem-solving style, not about acquiring new mathematical knowledge.

---

#### Group Relative Policy Optimization (GRPO)

GRPO is the paper's most significant algorithmic contribution. It is a variant of Proximal Policy Optimization (PPO) designed to reduce the memory and computational overhead of RL fine-tuning for language models while maintaining or improving effectiveness for mathematical reasoning tasks.

**Why PPO has high resource requirements.** To understand GRPO, one must first understand the component it eliminates: the value function (critic model). In standard PPO applied to LLMs, three models must be maintained in memory simultaneously:

1. The **policy model** (the LLM being trained, with parameters θ). This is the model that generates text.
2. The **reference model** (a frozen copy of the initial SFT model). This is used to compute a KL divergence penalty that prevents the policy from drifting too far from its initialization, which would cause reward hacking or language degradation.
3. The **value model** (a separate neural network of comparable size to the policy model, with parameters ψ). This is trained to predict the expected future reward from each token position, producing a baseline that is subtracted from actual rewards to compute the advantage.

The value model is the resource bottleneck: for a 7B-parameter policy model, the value model typically also has ~7B parameters, effectively doubling the GPU memory required for training. Additionally, the value model must be trained to accurately predict per-token advantages even though the reward signal in LLM applications typically only arrives at the final token (the answer is either correct or incorrect). This means the value model must learn to perform credit assignment—guessing which intermediate steps contributed to the final outcome—which is difficult and can introduce noise into the advantage estimates.

**The GRPO insight: replace the learned value function with a group-based baseline.** GRPO's key idea is simple: instead of training a separate value model to estimate the baseline for advantage computation, **use the average reward of multiple outputs generated for the same input as the baseline.** For each question q, GRPO samples a group of G outputs {o₁, o₂, ..., o_G} from the current policy, scores them all with the reward model, and normalizes the scores within the group. The normalized score for an output becomes the advantage for all tokens in that output.

This works because of a specific property of reward models used in mathematical reasoning: they are typically trained on **comparative data** (pairs of outputs where one is better than the other). The reward model learns to rank outputs, and its absolute scores are meaningful primarily in relative terms—Output A scoring 0.8 vs. Output B scoring 0.3 tells you that A is better than B, but the absolute numbers don't have a natural calibration. By normalizing within a group of outputs for the same question, GRPO effectively converts the reward model's comparative assessments into a zero-mean, unit-variance signal that serves as the advantage.

---

#### GRPO Objective Derivation

The GRPO objective builds directly on the PPO objective, so we trace the derivation from PPO to GRPO.

**PPO Objective (Equation 1).** The standard PPO surrogate objective for LLM fine-tuning is:

$$J_{PPO}(\theta) = \mathbb{E}_{q\sim P(Q), o\sim \pi_{\theta_{old}}(O|q)} \frac{1}{|o|} \sum_{t=1}^{|o|} \min\left( \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})} A_t, \text{clip}\left( \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})}, 1 - \varepsilon, 1 + \varepsilon \right) A_t \right)$$

where:

- $q \sim P(Q)$ is a question sampled from the dataset
- $o \sim \pi_{\theta_{old}}(O|q)$ is an output sampled from the old (pre-update) policy
- $|o|$ is the number of tokens in the output
- $\pi_\theta(o_t|q, o_{<t})$ is the probability assigned to token $o_t$ by the current policy (the one being optimized)
- $\pi_{\theta_{old}}(o_t|q, o_{<t})$ is the probability assigned to token $o_t$ by the old policy (the one that generated the output)
- $\varepsilon$ is a clipping hyperparameter (typically 0.1 or 0.2) that prevents the policy from changing too much in a single update
- $A_t$ is the **advantage** at token position $t$—a scalar indicating whether the action taken at that step was better or worse than expected

**What it computes:** For each token in each generated output, PPO computes the ratio of its probability under the current policy to its probability under the old policy. This ratio measures how much more (or less) likely the current policy makes that token choice compared to the policy that originally generated it. The advantage $A_t$ tells us whether that token choice was good (positive advantage) or bad (negative advantage). The objective encourages the policy to increase the probability of tokens with positive advantage and decrease the probability of tokens with negative advantage—but the clipping prevents the ratio from going outside $[1-\varepsilon, 1+\varepsilon]$, which limits how much the policy can change in a single update and stabilizes training.

**Why this form:** The clipping mechanism is the defining innovation of PPO. Without it, a single high-advantage token could cause the policy to increase its probability dramatically, potentially causing the policy to collapse to a deterministic mode and losing the exploration needed for further improvement. The clip creates a trust region: if the advantage is positive, the policy can increase the token's probability, but only up to a factor of $1+\varepsilon$ relative to the old policy. If the advantage is negative, the policy can decrease the probability, but only down to a factor of $1-\varepsilon$. This prevents destructive large updates while still allowing meaningful improvement.

**PPO Advantage Computation.** In standard PPO, the advantage $A_t$ is computed using Generalized Advantage Estimation (GAE, Schulman et al., 2015):

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(o_{t+1}) - V(o_t)$ is the temporal difference error, $r_t$ is the reward at step $t$, $V$ is the learned value function, $\gamma$ is a discount factor, and $\lambda$ is a trace-decay parameter. This computation requires the value function $V$ to be evaluated at every token position, which means the value model must be run forward through the entire sequence—doubling the forward-pass cost compared to just running the policy model.

**PPO Reward with KL Penalty (Equation 2).** In the LLM context, the per-token reward includes a KL penalty from the reference model:

$$r_t = r_\varphi(q, o_{\leq t}) - \beta \log \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{ref}(o_t|q, o_{<t})}$$

where:

- $r_\varphi(q, o_{\leq t})$ is the reward from a trained reward model (typically only non-zero at the final token, where it gives the answer correctness score)
- $\pi_{ref}$ is a frozen reference model (usually the initial SFT model)
- $\beta$ is the KL penalty coefficient
- The term $\beta \log \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{ref}(o_t|q, o_{<t})}$ penalizes the policy for diverging from the reference model at each token

**What it computes:** The effective reward for each token is the reward model's score minus a penalty for how much the current policy's per-token probabilities differ from the reference model's. The KL penalty prevents reward hacking—without it, the policy could learn to generate nonsensical text that happens to score highly under the reward model (e.g., outputting "The answer is 42" for every math problem and hoping 42 is correct).

**Why this form:** The KL penalty is added to the reward rather than directly to the loss because it needs to interact with the advantage computation. The advantage is computed from cumulative future rewards, so including the KL penalty in the reward signal ensures that the advantage already accounts for the cost of diverging from the reference model. An alternative would be to add a separate KL regularization term to the PPO objective, but this would require tuning an additional hyperparameter and could interact unpredictably with the clipping mechanism.

**GRPO Objective (Equation 3).** GRPO replaces the value function-based advantage with a group-based normalized reward and moves the KL penalty from the reward to the loss:

$$J_{GRPO}(\theta) = \mathbb{E}_{q\sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min\left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, \text{clip}\left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}, 1 - \varepsilon, 1 + \varepsilon \right) \hat{A}_{i,t} \right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right)$$

where the KL divergence term is defined using the following unbiased estimator (Schulman, 2020):

$$D_{KL}(\pi_\theta || \pi_{ref}) = \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1$$

**What is different from PPO.** Three key changes:

1. **Group sampling:** Instead of sampling one output per question, GRPO samples G outputs {o₁, ..., o_G} for each question. The expectation is now over groups of outputs rather than individual outputs.

2. **Group-based advantage $\hat{A}_{i,t}$:** The advantage for each token is computed from the relative rewards of the G outputs in its group, without any learned value function. The specific computation depends on whether outcome supervision or process supervision is used (detailed below).

3. **KL penalty in the loss, not the reward:** The KL divergence is added directly to the objective function rather than incorporated into the per-token reward. This simplifies the advantage computation—since $\hat{A}_{i,t}$ no longer needs to account for the KL penalty, it can be computed purely from the reward model's scores.

**Why this form.** The design decisions address specific practical problems with PPO:

- **Removing the value model** eliminates the memory overhead of maintaining a second large neural network. For a 7B parameter model, this roughly halves the GPU memory required for RL training. It also eliminates the noise introduced by imperfect value function approximation—the group-based baseline is a Monte Carlo estimate that, while potentially higher-variance than a learned value function for individual outputs, benefits from the fact that G outputs are always available and their mean is an unbiased baseline.

- **Moving the KL penalty to the loss** avoids complicating the advantage computation. In PPO, the advantage mixes the reward model signal with the KL penalty, making it difficult to disentangle whether a token's advantage is high because the answer was correct or because the policy stayed close to the reference. In GRPO, the KL penalty operates independently of the advantage, and its magnitude is controlled by the hyperparameter $\beta$ directly.

- **The KL divergence estimator** used (Equation 4) has the property of being guaranteed positive—unlike the simpler estimator $\log(\pi_{ref}/\pi_\theta)$, which can be negative. This is important because the KL penalty should always push the policy toward the reference; a negative KL "penalty" would push the policy away, which is counterproductive. The form $\frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1$ is a second-order Taylor approximation to the true KL divergence that is always non-negative.

---

#### Outcome Supervision with GRPO

Outcome supervision is the simpler variant of GRPO: the reward model provides a single scalar score for each complete output, and this score is normalized across the group to produce the advantage for all tokens in that output.

**Procedure.** For each question $q$:

1. Sample G outputs $\{o_1, o_2, ..., o_G\}$ from the old policy $\pi_{\theta_{old}}$.
2. Score each output with the reward model, yielding rewards $r = \{r_1, r_2, ..., r_G\}$.
3. Normalize the rewards by subtracting the group mean and dividing by the group standard deviation:

$$\tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

4. Set the advantage for every token in output $o_i$ to this normalized reward: $\hat{A}_{i,t} = \tilde{r}_i$ for all $t$.

**What this means operationally.** If an output scored higher than the group average, every token in that output gets a positive advantage—the policy is encouraged to make all the token choices that led to that output more likely. If an output scored lower than the group average, every token gets a negative advantage—the policy is discouraged from making those token choices. The magnitude of the advantage depends on how many standard deviations above or below the mean the output's score falls.

**Why normalization matters.** Normalizing by the group mean and standard deviation serves three purposes:

1. **Calibration:** Raw reward model scores may drift over training as the policy generates different output distributions. Normalization within each group ensures advantages are always zero-mean and unit-variance, providing a consistent training signal regardless of the absolute reward scale.

2. **Relative comparison:** Mathematical reasoning rewards are inherently comparative. A reward score of 0.6 means different things for an easy question (where the average output might score 0.8) versus a hard question (where the average might be 0.2). Normalization converts absolute scores into relative rankings within the group, which aligns with how reward models are trained (on comparative preference data).

3. **Variance reduction:** The group mean, while not as sophisticated as a learned value function, is an unbiased estimator of the expected reward for that question under the current policy. Subtracting it from each reward removes question-difficulty-dependent baselines, isolating the effect of the specific output rather than the inherent difficulty of the question.

**The limitation of outcome supervision.** By assigning the same advantage to all tokens, outcome supervision provides no per-step credit assignment. If a solution has 200 tokens and the first step is wrong but the final answer happens to be correct, all 200 tokens receive positive reinforcement—including the erroneous first step. This can lead to the policy learning spurious correlations rather than genuine reasoning improvements. This is the motivation for process supervision.

---

#### Process Supervision with GRPO

Process supervision addresses the credit assignment problem by providing rewards at the end of each reasoning step, not just at the final answer. This requires a **process reward model** that can evaluate intermediate solution steps.

**Procedure.** For each question $q$:

1. Sample G outputs $\{o_1, o_2, ..., o_G\}$ from the old policy.

2. Use a process reward model to score each step of each output. For output $i$, which has $K_i$ steps, the reward model produces rewards at the end token indices of each step: $\{r_i^{\text{index}(1)}, r_i^{\text{index}(2)}, ..., r_i^{\text{index}(K_i)}\}$, where $\text{index}(j)$ is the token position of the end of step $j$, and $r_i^{\text{index}(K_i)}$ is the reward at the final token (the answer).

3. Collect all step-level rewards across all G outputs into a single vector $\mathbf{R}$ and normalize globally:

$$\tilde{r}_i^{\text{index}(j)} = \frac{r_i^{\text{index}(j)} - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$$

Note the crucial difference from outcome supervision: the normalization is across *all steps in all outputs in the group*, not just across the final-answer rewards. This means a particularly good intermediate step can receive a high normalized reward even if the final answer is incorrect.

4. For each token $t$ in output $i$, compute the advantage as the sum of normalized rewards for all subsequent steps that follow token $t$:

$$\hat{A}_{i,t} = \sum_{\text{index}(j) \geq t} \tilde{r}_i^{\text{index}(j)}$$

**What this means operationally.** Each token's advantage is the sum of rewards for all reasoning steps that occur after that token. If token $t$ appears before step 3 of a 5-step solution, its advantage is the sum of the normalized rewards for steps 3, 4, and 5. A token that leads to a series of highly-rewarded subsequent steps receives a large positive advantage; a token that leads to poorly-rewarded steps receives a negative advantage.

**Why this form.** The cumulative sum structure reflects the causal nature of text generation: the consequence of generating a particular token is everything that comes after it. By summing the rewards of all subsequent steps, the advantage for token $t$ captures the expected future reward conditioned on having generated token $t$ and the preceding context. This is analogous to the return-to-go in reinforcement learning, but with the important distinction that rewards are only assigned at step boundaries (not at every token), and the "discount" is effectively 1.0—the reward for every subsequent step contributes equally, regardless of how far in the future it occurs.

The advantage of process supervision over outcome supervision is that it provides **differential credit within a solution**. If steps 1 and 2 are logically sound but step 3 introduces an error that leads to a wrong final answer, process supervision can assign positive advantages to tokens in steps 1 and 2 (because those steps' own normalized rewards are high) while assigning negative advantages to tokens in step 3. Outcome supervision, by contrast, would assign uniformly negative advantages to all tokens in an incorrect solution, potentially discouraging the model from making correct initial steps that simply happened to be followed by a later error.

**The process reward model training.** The paper follows the approach of Wang et al. (2023b) for training the process reward model, though the training details are not extensively described in the main text. The process reward model is initialized from DeepSeekMath-Base 7B and trained to predict step-level correctness. The training data consists of sampled solutions with per-step annotations indicating whether each step is on track toward the correct answer.

---

#### Iterative RL with GRPO

As RL training progresses, the policy model's output distribution shifts away from what the initial reward model was trained on. This distribution shift can cause the reward model to become miscalibrated—it was trained to evaluate outputs from the initial SFT model, but it is now evaluating outputs from a progressively more RL-tuned model.

Iterative GRPO (Algorithm 1) addresses this by periodically updating the reward model using fresh data sampled from the current policy model.

**Algorithm 1 detail.** The iterative procedure operates as follows:

1. **Initialization:** Start with the initial policy model $\pi_{\theta_{init}}$ (the SFT model) and the initial reward model $r_\varphi$.

2. **Outer loop (iterations 1 to I):**
   - Set the reference model $\pi_{ref}$ to the current policy model $\pi_\theta$. (This is a snapshot of the policy at the start of the iteration.)
   - **Inner loop (steps 1 to M):**
     - Sample a batch of questions $D_b$ from the training set $D$.
     - Set the old policy $\pi_{\theta_{old}}$ to the current policy $\pi_\theta$.
     - For each question in the batch, sample G outputs from $\pi_{\theta_{old}}$ and score them with the current reward model $r_\varphi$.
     - Compute advantages using group relative normalization (outcome or process supervision).
     - Perform $\mu$ GRPO update steps on the policy model $\pi_\theta$ using the computed advantages.
   - After the inner loop completes (M steps of policy updates), **update the reward model** $r_\varphi$ by training it on new data sampled from the updated policy model, using a replay mechanism that retains 10% of historical data.
   - Proceed to the next iteration, where the updated reward model will supervise the policy.

**The replay mechanism.** When retraining the reward model, 10% of the training data is retained from previous iterations, and 90% is fresh data from the current policy. The 10% replay buffer prevents catastrophic forgetting in the reward model—without it, the reward model could overfit to the current policy's idiosyncratic error patterns and lose the ability to evaluate a broader range of solution styles.

**Why iterative RL is necessary.** The paper's experiments with iterative RL (Figure 6) show that a second iteration of RL provides additional gains over a single iteration, particularly on MATH (approximately 49% → 51.7%). This suggests that the initial reward model, even when trained on SFT-model outputs, becomes a bottleneck after the policy has been substantially improved: the policy is now generating outputs that are systematically different from what the reward model was trained to evaluate, and the reward model's scores may not accurately reflect true quality. By retraining the reward model on policy-generated data, the system creates a virtuous cycle: better policy → higher-quality training data for the reward model → more accurate reward signal → further policy improvement.

---

#### Training Configuration for DeepSeekMath-RL

**Training data.** The RL training uses only chain-of-thought-format questions related to GSM8K and MATH from the SFT data, totaling approximately 144K questions. This is deliberately a subset of the SFT data (which has 776K examples total). The authors choose this restriction to "investigate the impact of RL on benchmarks that lack data throughout the RL phase"—in other words, to test whether RL generalizes to out-of-domain benchmarks. GSM8K and MATH with chain-of-thought reasoning are considered **in-domain** tasks; all other benchmarks (CMATH, Gaokao, MGSM-zh, tool-integrated reasoning, etc.) are considered **out-of-domain**.

**Reward model training.** The initial reward model is trained from DeepSeekMath-Base 7B (not from the Instruct model) with a learning rate of 2e-5. Training follows the protocol of Wang et al. (2023b), though specific training data construction details are not given in the paper beyond the citation.

**GRPO hyperparameters:**
- Policy model learning rate: 1e-6 (very low, typical for RL fine-tuning where the model should change gradually)
- KL penalty coefficient β: 0.04 (determines the strength of the regularizer pulling the policy toward the reference model)
- Number of outputs sampled per question (G): 64
- Maximum output length: 1024 tokens
- Training batch size: 1024 (effective questions per batch; with 64 outputs per question, the total number of outputs per batch is 1024 × 64 = 65,536)
- Policy updates per exploration stage (μ): 1 (the policy is updated only once after each round of sampling—this is the simplest setting that prevents the policy from changing too much before getting fresh exploration data)

**Why these choices.** The single-update-per-exploration setting (μ = 1) simplifies the analysis in Appendix A.1. With μ = 1, $\pi_{\theta_{old}} = \pi_\theta$ before the update, so the clipping mechanism in PPO/GRPO never activates—the ratio is always 1.0 for the data that was just sampled. This makes the gradient analysis cleaner (Equations 16, 19), but in principle, multiple updates per exploration stage could be more sample-efficient. The choice of G = 64 means that the group baseline is computed from 64 independent outputs for each question, which provides a reasonably low-variance estimate of the expected reward.

---

#### The Unified RL Paradigm (Equation 5)

Section 5.2.1 presents a unified analytical framework that expresses several seemingly distinct training methods (SFT, RFT, DPO, PPO, GRPO) as variants of the same underlying form. The general gradient with respect to the policy parameters θ is:

$$\nabla_\theta \mathcal{J}_A(\theta) = \mathbb{E}_{(q, o) \sim \mathcal{D}} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} GC_A(q, o, t, \pi_{rf}) \nabla_\theta \log \pi_\theta(o_t | q, o_{<t}) \right]$$

where:

- $\mathcal{D}$ is the **data source**—the distribution over (question, output) pairs used for training
- $\pi_{rf}$ is the **reward function**—the source of the training signal
- $GC_A$ is the **gradient coefficient** produced by algorithm A—a scalar at each token position that determines the magnitude and sign of the update

**What this equation says.** Every training method for autoregressive language models can be understood as doing the same basic operation: for each token in each training output, compute a scalar gradient coefficient, and use it to scale the gradient of the log-probability of that token. If $GC_A > 0$, the update increases the probability of that token in that context (reinforcement). If $GC_A < 0$, the update decreases the probability (penalization). If $GC_A = 0$, the token is ignored. The methods differ only in (a) where the training data comes from, (b) what reward function provides the signal, and (c) how the algorithm transforms the reward signal into the gradient coefficient.

**Why this unification matters.** This framework makes precise what would otherwise be vague comparisons. "RFT and Online RFT differ only in data source" is a precise statement: both use rule-based reward (answer correctness) and have the same gradient coefficient function (Equation 10: $GC_{RFT} = \mathbb{I}(o)$, which is 1 if the answer is correct and 0 otherwise), but RFT samples outputs from the frozen SFT model while Online RFT samples from the current policy model. The performance difference between them (Figure 5 shows Online RFT substantially outperforms RFT) can therefore be attributed entirely to the data source axis—online sampling is better because it provides training data that reflects the current policy's actual output distribution.

Similarly, "Online RFT and GRPO differ only in the gradient coefficient" (when both use online data and rule-based reward) tells us that GRPO's superior performance over Online RFT must be due to its more sophisticated transformation of the reward signal: Online RFT assigns gradient coefficient 1 to all correct outputs (uniform reinforcement) and 0 to incorrect outputs (no penalization), while GRPO assigns continuous-valued gradient coefficients that can be positive or negative, depending on how much better or worse than the group average the output scored.

**Gradient coefficients for key methods (derived in Appendix A.1):**

- **SFT:** $GC = 1$ for all tokens in all examples. The data source is human-curated SFT data; there is no reward function—all selected examples are treated as equally correct.
- **RFT / Online RFT:** $GC = \mathbb{I}(o) = 1$ if the answer is correct, 0 otherwise. Incorrect outputs are simply not used for training.
- **DPO:** $GC$ depends on the log-ratio of policy and reference probabilities for both the chosen and rejected outputs (Equation 14). Correct and incorrect outputs both contribute to the gradient, but with opposite signs.
- **PPO:** $GC = A_t$, where $A_t$ is computed via GAE from reward model scores and the learned value function (Equation 18).
- **GRPO:** $GC = \hat{A}_{i,t} + \beta(\frac{\pi_{ref}}{\pi_\theta} - 1)$, where $\hat{A}_{i,t}$ is the group-normalized advantage (Equation 21).

**The critical distinction: Online RFT cannot penalize incorrect outputs.** The gradient coefficient for Online RFT (Equation 10) is zero for all tokens in incorrect outputs. This means the policy receives no gradient signal from its mistakes—it only learns from successes. GRPO's gradient coefficient, by contrast, can be negative for outputs that score below the group mean, actively reducing the probability of the token choices that led to those low-scoring outputs. The paper argues (and Figure 5 empirically supports) that this ability to learn from both successes and failures is a key advantage of GRPO over rejection sampling approaches.

---

#### Why RL Works: Maj@K vs. Pass@K (Section 5.2.2)

To understand *why* GRPO improves benchmark scores, the paper evaluates two distinct measures of model capability:

- **Pass@K:** The probability that at least one of K independent samples from the model is correct. This measures the model's *fundamental capability*—its ability to ever produce a correct solution, even if it doesn't do so consistently.

- **Maj@K:** The accuracy of majority voting over K independent samples (i.e., generating K solutions, taking the most common final answer, and checking if it is correct). This measures the model's *output distribution robustness*—how consistently the correct answer appears among the top candidates.

**Figure 7 results.** On both GSM8K and MATH:

- **Pass@K is essentially unchanged between the Instruct and RL models.** At K=64 on MATH, Pass@K is approximately 78% for both Instruct and RL models. This means RL did not teach the model to solve fundamentally new types of problems—the set of problems where at least one of 64 samples is correct is roughly the same for both models.

- **Maj@K improves substantially with RL.** At K=64 on MATH, Maj@K increases from approximately 60% (Instruct) to nearly 75% (RL). This means that while the RL model doesn't produce correct solutions for new problems, it produces correct solutions *more consistently* for problems it could already solve—the correct answer is more likely to be the most common one in a set of 64 samples.

**Interpretation.** The paper interprets this as evidence that RL improves the *ranking* of correct solutions within the model's output distribution, not the *existence* of correct solutions. In the SFT model, correct reasoning paths exist (as evidenced by Pass@K), but they are often assigned lower probability than incorrect but plausible-sounding paths. The model "knows" the right answer but doesn't consistently output it. RL, by penalizing incorrect outputs and reinforcing correct ones, redistributes probability mass so that correct reasoning paths become more likely relative to incorrect ones.

This connects to the "misalignment problem" identified by Wang et al. (2023a): SFT models suffer from a mismatch between generation probability and answer correctness. The model might be able to produce a correct chain-of-thought, but it places higher probability on an incorrect but superficially similar chain. RL corrects this misalignment by making the generation probability better correlated with actual answer correctness.

**Implications.** This finding has two important implications:

1. **RL cannot create new capabilities.** If a problem is fundamentally beyond the base model's reach (Pass@1 ≈ 0), RL will not help—there are no correct solutions to reinforce. This implies that RL for reasoning is fundamentally bounded by the base model's capability frontier, which is set during pre-training.

2. **RL is most valuable for models that already have non-trivial Pass@K.** The larger the gap between Pass@K and Maj@K, the more room there is for RL to improve downstream accuracy. This suggests that pre-training should aim to maximize Pass@K (the model's latent capability), and RL should be used to convert that latent capability into consistent correct outputs.

## 4. Key Insights and Innovations

### Innovation 1: Web-Scale Math Data Selection as an Iterative, Human-Guided Classification Problem

The dominant approach to building mathematical pre-training corpora before DeepSeekMath was either one-shot heuristic filtering (e.g., OpenWebMath's classifier applied to Common Crawl once) or reliance on curated sources like arXiv and textbooks (e.g., MathPile, Proof-Pile-2). The implicit assumption was that mathematical content on the web is either already well-separated from noise or so diffuse that precise extraction requires prohibitive effort.

DeepSeekMath fundamentally reframes this as an **active learning problem at web scale**. The key conceptual move is recognizing that a classifier trained only on a seed corpus (OpenWebMath) has *unknown blind spots* — it will miss mathematical content that doesn't resemble the seed's style, domain distribution, or formatting conventions. Rather than accepting these blind spots as inevitable, the paper introduces a **domain-level feedback loop**: identify clusters of the web (domains) where the classifier appears to be making false-negative errors, solicit minimal human input (URL path annotation), enrich the seed corpus, and retrain. This is not merely "more data" — it is a qualitatively different data collection paradigm that treats the classifier as a continuously improving instrument rather than a one-shot filter.

The significance of this reframing extends beyond mathematics. Any domain where high-quality content exists on the web but is interleaved with noise — law, medicine, chemistry, programming — could apply the same iterative pipeline. The paper explicitly notes this in Section 2.1: "It's worth noting that this approach is also applicable to other domains, such as coding." This positions the data collection methodology as a generalizable contribution, not just a math-specific engineering effort.

The empirical evidence supporting this innovation is in Figure 3 and Table 1: the DeepSeekMath Corpus-trained model shows a steeper learning curve and more sustained improvements than all baselines, including Proof-Pile-2 which is nearly half its size. At 50B tokens of training (one epoch of Proof-Pile-2), the DeepSeekMath Corpus model already outperforms Proof-Pile-2 across benchmarks, confirming that the iterative pipeline produces inherently higher-quality data, not just more of it. The convergence criterion — stopping at four iterations when 98% of data was already collected in iteration three — provides a principled termination condition that avoids indefinite annotation cost.

This is a **fundamental methodological contribution** with implications well beyond the specific model: it provides a replicable recipe that the research community can apply to any domain where a small seed corpus of high-quality web content exists. The 7-9× scale advantage over prior math web corpora (120B vs. ~14-15B tokens) is a consequence of the method, not the method itself.

---

### Innovation 2: The arXiv Negative Result — Challenging a Widely-Held Assumption

Nearly every major mathematical language model prior to DeepSeekMath — Minerva (Lewkowycz et al., 2022a), Llemma (Azerbayev et al., 2023), and the MathPile corpus (Wang et al., 2023c) — included arXiv papers as a significant component of their pre-training data. The intuition was straightforward: arXiv contains the world's mathematical research literature in LaTeX format; training on it should expose models to advanced mathematical notation, proof structures, and theorem statements. This assumption was so widely accepted that arXiv inclusion was effectively a default design choice, rarely questioned in published work.

DeepSeekMath challenges this assumption with controlled experiments at two model scales (1.3B and 7B) across two arXiv corpora (MathPile and ArXiv-RedPajama), consistently finding **no improvement or outright degradation** on every mathematical benchmark tested. GSM8K drops from 29.0% to 23.6% when training DeepSeek-Coder-Base-v1.5 on MathPile; MATH shows negligible change (12.5% vs. 11.5%); and formal theorem proving on miniF2F degrades from 20.1% to 16.8% on the validation set (Tables 8 and 9).

What makes this a genuine insight rather than just a failed experiment is the **diagnostic clarity** it provides. The paper doesn't simply report "arXiv doesn't work" — it implicitly identifies the distribution mismatch that explains why. arXiv papers are expository: they present completed mathematical results in a proof-theoretic style aimed at domain experts. Mathematical benchmarks, by contrast, test problem-solving: given a novel problem statement, produce a step-by-step solution. The format, cognitive demand, and discourse structure are fundamentally different. Training on arXiv teaches a model to *explain mathematics that has already been solved*; benchmarks test whether it can *solve mathematics it hasn't seen*. This distinction, while intuitive in retrospect, was not operationalized in prior work, which treated "mathematical content" as a unitary category.

This is a **negative result with significant implications**. It is rare in the ML literature for a paper to systematically test and reject a widely-held assumption, and rarer still for that rejection to come with actionable guidance (the paper still includes 10% arXiv in the final mixture, hedging that arXiv might help in combination or at larger scales, but the burden of proof has now shifted). Future work on mathematical language models must either justify arXiv inclusion with empirical evidence on their specific benchmarks or acknowledge that the prevailing wisdom is unsupported. The paper's careful caveating — noting that arXiv might help on unevaluated tasks like theorem informalization, or at larger model scales — models good scientific practice: the conclusion is strong within its tested scope but doesn't overclaim universality.

---

### Innovation 3: GRPO as a Resource-Efficient RL Algorithm with Conceptual Advantages Over PPO

Reinforcement learning for language model fine-tuning, as practiced in prior work (Ouyang et al., 2022; Luo et al., 2023; Wang et al., 2023b), relies on Proximal Policy Optimization (PPO) with a learned value function. This introduces a **practical barrier**: the value model is typically comparable in size to the policy model, roughly doubling GPU memory requirements and adding significant computational overhead. For a 7B-parameter model, this means effectively needing the resources to train a ~14B-parameter system. This barrier limits who can do RL fine-tuning and at what scale.

GRPO's surface-level contribution — eliminating the value model — is an engineering efficiency improvement. But the deeper conceptual move is recognizing that the value model is not merely expensive but **potentially misaligned with how reward models operate in the LLM context**. Reward models for mathematical reasoning are trained on comparative data (pairs of outputs ranked by quality). Their absolute scores are meaningful primarily in relative terms: output A scoring 0.8 vs. output B scoring 0.3 tells you A is preferable, but the numbers lack natural calibration. PPO's learned value function must predict these absolute scores, which creates a tension: the value function is trying to estimate an absolute quantity that is intrinsically comparative.

GRPO resolves this tension by computing the baseline *directly from the group of outputs that the reward model is already comparatively evaluating*. For each question, the reward model scores G outputs, and the group mean becomes the baseline — no separate value model needed. The normalization step (subtracting the mean, dividing by standard deviation) converts comparative reward model scores into a zero-mean, unit-variance advantage signal. This is not just cheaper; it is potentially *more principled* because the baseline comes from the same comparative process that the reward model was trained to perform.

The empirical evidence (Figure 5) shows that GRPO outperforms Online RFT, which uses the same data source (online sampling) but lacks continuous gradient coefficients. This confirms that the algorithmic innovation (continuous-valued, potentially negative gradient coefficients) is independently valuable beyond the data source improvement. The process supervision variant (GRPO+PS) further outperforms outcome supervision (GRPO+OS), demonstrating that the framework accommodates fine-grained credit assignment without architectural changes.

This is primarily a **practical innovation with conceptual depth**. It is not theoretically novel in the sense of introducing new mathematical frameworks — the use of group-based baselines for variance reduction is well-known in RL. But applying this idea to *replace the entire value function in LLM fine-tuning*, and recognizing that the comparative nature of LLM reward models makes group baselines particularly natural, represents a genuine synthesis of existing ideas into a more elegant and accessible whole. The reduction in resource requirements (roughly halving memory for the non-policy components of training) lowers the barrier to entry for RL fine-tuning, which could accelerate research in this area.

---

### Innovation 4: A Unified Gradient-Based Framework for Understanding Post-Training Methods

The literature on improving language models after pre-training has produced a proliferation of named methods — SFT, RFT, DPO, PPO, RLHF, and now GRPO — each with its own objective function, training procedure, and claimed advantages. Practitioners face a difficult question: given a task and a compute budget, which method should they use? The default approach has been empirical trial-and-error, guided by heuristics and community lore rather than principled analysis.

The unified paradigm in Section 5.2.1 and Appendix A.1 makes a significant conceptual contribution by showing that all these methods can be expressed as instances of a single gradient expression (Equation 5) that varies along exactly three axes: **Data Source** (where the training (question, output) pairs come from), **Reward Function** (what signal provides the training target), and **Gradient Coefficient** (how the algorithm transforms the reward signal into per-token update magnitudes). This is a **taxonomic innovation** — it doesn't introduce a new method but rather provides a lens for understanding existing ones.

The power of this framework is in the precise comparisons it enables. The paper can now say: "RFT and Online RFT differ only in Data Source" (offline vs. online sampling), and since Online RFT outperforms RFT (Figure 5), the performance gap is attributable entirely to the advantage of training on fresh policy samples rather than stale SFT-model samples. Similarly, "Online RFT and GRPO differ only in Gradient Coefficient" (binary vs. continuous-valued), and since GRPO outperforms Online RFT, the gradient coefficient's ability to penalize incorrect outputs differentially is causal. These are not vague qualitative claims — they follow directly from the framework's decomposition.

The framework also reveals blind spots. It shows that RFT and Online RFT *cannot penalize incorrect outputs* — their gradient coefficient is zero for all tokens in wrong answers (Equation 10). This is a structural limitation, not a parameter choice, and it explains why rejection sampling approaches plateau: they only learn from successes, never from failures. DPO, by contrast, *can* penalize incorrect outputs but uses a fixed gradient coefficient that doesn't adapt to varying degrees of correctness. PPO and GRPO introduce continuous gradient coefficients that can provide stronger reinforcement for better outputs and weaker (or negative) reinforcement for worse ones.

This is a **conceptual advance** rather than an empirical one. It does not directly improve benchmark scores, but it provides the intellectual scaffolding for understanding *why* certain methods work and for predicting *which* method will be most effective in a given scenario. The framework's identification of three independent axes suggests a research program: systematically explore the space of (Data Source, Reward Function, Gradient Coefficient) combinations, rather than treating each named method as a monolithic entity. This moves the field from recipe-following toward principled method design.

---

### Innovation 5: Diagnostic Separation of RL Gains into Distributional Robustness vs. Capability Expansion

It is tempting to interpret improved benchmark scores after RL as evidence that the model "got better at math" — that it learned new reasoning strategies or acquired deeper conceptual understanding. The paper provides a clean empirical refutation of this interpretation through the Pass@K vs. Maj@K analysis in Section 5.2.2 and Figure 7.

The critical finding: **GRPO improves Maj@K (majority voting accuracy) substantially but leaves Pass@K (the probability that at least one of K samples is correct) essentially unchanged**. On MATH at K=64, Maj@K jumps from ~60% (Instruct) to ~75% (RL), while Pass@K stays at ~78% for both models. This is a remarkably clean result — it isolates two different senses of "model capability" that are often conflated:

- **Fundamental capability (Pass@K):** Can the model ever produce a correct solution? This is determined by pre-training and reflects the model's latent knowledge and reasoning capacity.
- **Output distribution quality (Maj@K):** When the model generates multiple solutions, is the correct answer consistently ranked among the most probable? This is determined by the alignment between the model's generation probabilities and actual answer correctness.

The diagnosis is that RL improves the second but not the first. The SFT model already "knows" how to solve most problems where Pass@K > 0; it just doesn't consistently output those solutions because its probability distribution over reasoning paths is misaligned with correctness — plausible-sounding incorrect paths receive higher probability than the correct ones. RL corrects this misalignment by penalizing incorrect reasoning paths and reinforcing correct ones, redistributing probability mass without expanding the set of reachable correct solutions.

This is a **diagnostic insight** with practical and philosophical implications. Practically, it means RL for mathematical reasoning is fundamentally bounded by the base model's Pass@K — if a problem has near-zero chance of being solved by the base model, RL cannot help. Resources for improving a model should therefore prioritize pre-training data quality and scale (which determine Pass@K) before investing in RL (which converts latent capability into consistent performance). Philosophically, it reframes what "improvement" means: RL doesn't make the model smarter; it makes it more reliable at expressing the intelligence it already has.

The paper connects this to the "misalignment problem" identified by Wang et al. (2023a), but DeepSeekMath provides the cleanest empirical demonstration to date because it isolates the effect in a single model family with controlled pre-training and RL stages. The result also explains an otherwise puzzling pattern in the literature: why some RL-enhanced models show dramatic benchmark improvements while others don't. If the base model already has high Pass@K but low Maj@K (large misalignment), RL will produce large gains. If Pass@K is already near Maj@K (little misalignment), RL will help minimally. This reframes RL effectiveness as a function of the base model's output distribution characteristics, not an intrinsic property of the algorithm.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on a comprehensive suite of benchmarks spanning multiple mathematical domains and difficulty levels. English mathematical reasoning benchmarks include **GSM8K** (Cobbe et al., 2021, grade-school math word problems), **MATH** (Hendrycks et al., 2021, competition-level problems across algebra, probability, number theory, calculus, and geometry), **SAT** (Azerbayev et al., 2023), **OCW Courses** (Lewkowycz et al., 2022a), and **MMLU-STEM** (Hendrycks et al., 2020, multiple-choice STEM questions). Chinese benchmarks include **MGSM-zh** (Shi et al., 2023), **CMATH** (Wei et al., 2023), **Gaokao-MathCloze**, and **Gaokao-MathQA** (Zhong et al., 2023). Formal mathematics is evaluated using **miniF2F** (Zheng et al., 2021) with Isabelle as the proof assistant (Jiang et al., 2022). Natural language understanding and reasoning are assessed via **MMLU** (Hendrycks et al., 2020, 57 tasks) and **BBH** (Suzgun et al., 2022, 23 challenging multi-step reasoning tasks), with code generation evaluated on **HumanEval** (Chen et al., 2021) and **MBPP** (Austin et al., 2021). The specific split from Lightman et al. (2022) is used for MATH, but other benchmark splits are not explicitly specified.

- **Base model(s).** The paper primarily evaluates **DeepSeekMath-Base 7B**, initialized from DeepSeek-Coder-Base-v1.5 7B (Guo et al., 2024) and trained for 500B additional tokens on the data mixture described in Section 2.3. For corpus quality ablation experiments (Section 2.2), a smaller **DeepSeek-LLM 1.3B** model is used as a cost-effective proxy. For reinforcement learning experiments, **DeepSeekMath-Instruct 7B** (the SFT version) serves as the base for GRPO training, producing **DeepSeekMath-RL 7B**. The choice of 7B scale is deliberate: it sits in a regime where competitive mathematical reasoning is achievable with modest compute, making the model practically deployable while leaving room for test-time strategies to demonstrate meaningful gains.

- **Metrics.** The primary metric across all mathematical benchmarks is **accuracy** — the fraction of problems for which the model's final answer matches the ground truth. For chain-of-thought reasoning, the model generates a self-contained text solution and the final answer is extracted and compared to the reference. For program-of-thought and tool-integrated reasoning, the model writes a Python program; the execution result is evaluated as the answer. For multiple-choice benchmarks (MMLU-STEM, Gaokao-MathQA), accuracy is measured as the fraction of correctly selected options. For formal theorem proving on miniF2F, success is measured by whether the model-generated proof sketch, when combined with Sledgehammer (Paulson, 2010), produces a valid Isabelle proof. Pass@K and Maj@K are additionally reported in Section 5.2.2: Pass@K measures whether at least one of K independent samples is correct, while Maj@K measures the accuracy of majority voting over K samples. All results are reported as percentages.

- **Baselines.** For base model comparisons (Table 2), the paper benchmarks against **Minerva** at 7B, 62B, and 540B scales (Lewkowycz et al., 2022a, closed-source), **Mistral 7B** (Jiang et al., 2023, open-source general model), and **Llemma 7B and 34B** (Azerbayev et al., 2023, open-source math-trained models). For instruction-tuned comparisons (Table 5), baselines include closed-source models (**Gemini Ultra and Pro** from Anil et al., 2023; **GPT-4** and **GPT-4 Code Interpreter** from OpenAI, 2023; **Inflection-2**; **GPT-3.5**; **Grok-1**; **Baichuan-3**; **GLM-4**) and open-source models (**InternLM2-Math 20B**, **Qwen 72B** from Bai et al., 2023, **Math-Shepherd-Mistral 7B** from Wang et al., 2023b, **WizardMath-v1.1 7B and v1.0 70B** from Luo et al., 2023, **DeepSeek-LLM-Chat 67B** from DeepSeek-AI, 2024, **MetaMath 70B** from Yu et al., 2023, **SeaLLM-v2 7B** from Nguyen et al., 2023, **ChatGLM3 6B**, **ToRA 34B** from Gou et al., 2023, and **MAmmoTH 70B** from Yue et al., 2023). For corpus quality experiments (Table 1), baselines are models trained on **MathPile** (Wang et al., 2023c), **OpenWebMath** (Paster et al., 2023), and **Proof-Pile-2** (Azerbayev et al., 2023).

- **Generation budget / compute accounting.** Pre-training compute is measured in **tokens processed**, with the 1.3B ablation models trained for 150B tokens and DeepSeekMath-Base 7B trained for 500B tokens. Batch sizes are specified in tokens (4M for 1.3B, 10M for 7B). For RL experiments, compute is measured in the number of sampled outputs per question (G = 64) and the number of training steps (M policy updates per iteration, with I = 2 outer-loop iterations reported in Figure 6). RL training data is restricted to ~144K chain-of-thought questions from GSM8K and MATH. No FLOPs-matched comparison between pretraining and inference compute is performed — this is a distinction from the reference example paper's approach.

- **Cross-validation / statistical protocol.** The paper does not employ formal cross-validation for the main results. For the corpus quality experiments (Table 1), models are trained once on each corpus and evaluated on standard benchmark test sets. For SFT and RL, models are trained on the constructed training sets and evaluated on held-out benchmark test sets. The iterative RL experiments (Figure 6) report multiple iterations, but these are sequential refinements on the same training data distribution rather than cross-validation folds. No confidence intervals or statistical significance tests are reported for any result. The MATH benchmark uses 500 test questions (from Lightman et al., 2022), which provides reasonable but not large-sample statistical power — a ~2 percentage point difference represents roughly 10 questions, which could arise from sampling variance.

### Main Quantitative Results

#### Corpus Quality Validation (1.3B Scale)

The paper first validates the DeepSeekMath Corpus against existing mathematical corpora using DeepSeek-LLM 1.3B trained for 150B tokens on each corpus. Table 1 presents the headline comparison across 8 benchmarks.

**DeepSeekMath Corpus substantially outperforms all prior corpora.** On GSM8K, the DeepSeekMath Corpus-trained model achieves 23.8%, compared to 14.3% for Proof-Pile-2 (the next best), 11.5% for OpenWebMath, and 2.7% for MathPile — representing a 9.5 percentage point absolute improvement over the previous state-of-the-art corpus. On MATH, the lead is smaller but consistent: 13.6% vs. 11.2% for Proof-Pile-2 and 8.9% for OpenWebMath. The most dramatic advantage appears on Chinese benchmarks: CMATH reaches 41.5% with DeepSeekMath Corpus vs. 19.9% for Proof-Pile-2 and 16.8% for OpenWebMath, while Gaokao-MathQA achieves 23.6% vs. 11.7% and 14.2% respectively. This multilingual advantage is explicitly attributed to DeepSeekMath Corpus containing both English and Chinese mathematical content, whereas prior corpora were English-centric.

**Training on MathPile degrades performance relative to no math training.** Table 1 reveals that MathPile training produces lower scores than "No Math Training" on 6 of 8 benchmarks: GSM8K drops from 2.9% to 2.7%, CMATH collapses from 12.3% to 1.2%, and MMLU-STEM drops from 19.5% to 15.7%. This is a striking negative result: adding MathPile data is actively harmful for mathematical reasoning. The paper does not deeply analyze the mechanism, but the implication is that arXiv-heavy corpora (MathPile is >85% arXiv) introduce a distribution shift that interferes with the model's existing reasoning capabilities without providing compensatory benefits.

**Learning curves show quality and scale advantages.** Figure 3 plots benchmark performance as a function of training tokens for each corpus. The DeepSeekMath Corpus curve shows a consistently steeper slope than Proof-Pile-2 across all training budgets, and continues to improve beyond the point where Proof-Pile-2 plateaus (at roughly 50B tokens, corresponding to one epoch of Proof-Pile-2's 51.9B-token corpus). The paper explicitly notes this: "the model trained on the DeepSeekMath Corpus demonstrates better performance than Proof-Pile-2 at 50B tokens (1 full epoch of Proof-Pile-2), indicating the average quality of DeepSeekMath Corpus is higher." This is an important distinction: DeepSeekMath Corpus's advantage is not merely that it is larger (120B vs. 52B tokens), but that it is *higher quality per token*, enabling faster learning and sustained improvement.

#### DeepSeekMath-Base 7B: Comparison to Prior Base Models

Table 2 presents the main base model results, comparing DeepSeekMath-Base 7B against open-source models (Mistral 7B, Llemma 7B and 34B) and the closed-source Minerva family (7B, 62B, 540B) across 8 benchmarks using few-shot chain-of-thought prompting.

**DeepSeekMath-Base 7B surpasses all open-source base models by substantial margins.** On GSM8K, DeepSeekMath-Base achieves 64.2%, compared to 54.0% for Llemma 34B (10.2 percentage point improvement despite being ~5× smaller) and 40.3% for Mistral 7B. On MATH, the gap is even more pronounced: 36.2% vs. 25.3% for Llemma 34B (10.9 percentage points) and 14.3% for Mistral 7B. The paper describes this as "surpass[ing] existing open-source base models by over 10% absolute" on MATH, which is accurate.

**DeepSeekMath-Base 7B outperforms Minerva 540B on several benchmarks despite being ~77× smaller.** On GSM8K, DeepSeekMath-Base's 64.2% exceeds Minerva 540B's 58.8%. On MATH, 36.2% beats Minerva 540B's 33.6%. On OCW Courses, DeepSeekMath-Base's 15.4% is competitive with Minerva 540B's 17.6% and exceeds Minerva 62B's 12.0%. On MMLU-STEM, DeepSeekMath-Base's 56.5% trails Minerva 540B's 63.9% by a meaningful margin, but still outperforms Minerva 7B (35.6%) and is competitive with Minerva 62B (53.9%). The paper states that "a smaller model pre-trained on high-quality data could achieve strong performance as well," which is empirically supported, though the claim of "comparable performance" with Minerva 540B should be qualified: DeepSeekMath-Base leads on 3 of 5 common English benchmarks but trails on MMLU-STEM and OCW.

**Chinese benchmark performance is dramatically better than English-centric prior work.** On CMATH, DeepSeekMath-Base achieves 71.7%, compared to Llemma 34B's 56.1% and Mistral 7B's 44.9%. On Gaokao-MathQA, the lead is 35.3% vs. 26.2% (Llemma 34B) and 23.4% (Mistral 7B). This is attributed to the multilingual nature of DeepSeekMath Corpus, which naturally includes Chinese mathematical web pages alongside English ones.

#### DeepSeekMath-Base 7B: Mathematical Problem Solving with Tools

Table 3 evaluates program-aided mathematical reasoning (few-shot program-of-thought prompting) and formal theorem proving (informal-to-formal proving on miniF2F).

**Program-aided reasoning shows strong tool-use capability.** On GSM8K+Python, DeepSeekMath-Base achieves 66.9%, slightly surpassing Llemma 34B (64.6%) and substantially exceeding CodeLlama 34B (52.7%) and Mistral 7B (48.5%). On MATH+Python, the score of 31.4% exceeds Llemma 34B's 26.3% by 5.1 percentage points. The paper hypothesizes that code pre-training (DeepSeekMath-Base is initialized from a code model) enhances tool-use mathematical reasoning, which is supported by the ablation in Table 6 showing code training boosts program-aided math performance.

**Formal theorem proving performance is strong but the absolute numbers are modest.** On miniF2F-valid, DeepSeekMath-Base achieves 25.8%, compared to 21.0% for Llemma 34B and 20.6% for Llemma 7B. On miniF2F-test, 24.6% beats Llemma 34B's 21.3%. These are improvements of roughly 3-5 percentage points, but the absolute scores remain below 30%, indicating that formal theorem proving remains challenging for 7B-scale models even with strong mathematical pre-training.

#### DeepSeekMath-Base 7B: General Capabilities (Language Understanding, Reasoning, Code)

Table 4 evaluates the base model on MMLU (language understanding), BBH (reasoning), HumanEval and MBPP (code generation).

**Math pre-training improves language understanding and reasoning.** Comparing DeepSeekMath-Base to its precursor DeepSeek-Coder-Base-v1.5 (which was trained on code but not math): MMLU improves from 49.1% to 54.9% (5.8 percentage points), and BBH improves from 55.2% to 59.5% (4.3 percentage points). This supports the claim that "math pre-training benefits both language understanding and reasoning performance." The mechanisms are not analyzed, but the paper frames this as evidence that mathematical reasoning capability transfers to general reasoning tasks.

**Code performance is partially maintained but degrades.** HumanEval drops from 43.2% (DeepSeek-Coder-Base-v1.5) to 40.9% (DeepSeekMath-Base), and MBPP drops from 60.4% to 52.6%. The paper notes that "by including code tokens for continual training, DeepSeekMath-Base 7B effectively maintains the performance of DeepSeek-Coder-Base-v1.5 on the two coding benchmarks," but the MBPP drop of 7.8 percentage points suggests only partial mitigation of catastrophic forgetting. Despite this degradation, DeepSeekMath-Base still significantly outperforms the general model Mistral 7B on all three reasoning and coding benchmarks (MMLU: 54.9% vs. 62.4% — wait, Mistral is actually higher on MMLU; BBH: 59.5% vs. 55.7%; HumanEval: 40.9% vs. 28.0%; MBPP: 52.6% vs. 41.4%). The paper's claim of "significantly outperforms the general model Mistral 7B on the three reasoning and coding benchmarks" is accurate for reasoning (BBH) and coding (HumanEval, MBPP) but not for language understanding (MMLU), where Mistral 7B leads by 7.5 points.

#### DeepSeekMath-Instruct 7B: Instruction-Tuned Model Performance

Table 5 presents the main instruction-tuned results, comparing DeepSeekMath-Instruct 7B against a large set of open-source and closed-source models on chain-of-thought reasoning and tool-integrated reasoning benchmarks.

**Chain-of-thought reasoning: DeepSeekMath-Instruct dominates open-source models.** On GSM8K, DeepSeekMath-Instruct achieves 82.9% (Top1, no majority voting), compared to WizardMath-v1.1 7B's 83.2%, Math-Shepherd-Mistral 7B's 84.1%, and DeepSeek-LLM-Chat 67B's 84.1% — all of which are within 1.2 percentage points. The paper's claim that it "beats all open-source models from 7B to 70B" is therefore not strictly true for GSM8K Top1, where several models score marginally higher. However, the MATH results are decisive: 46.8% vs. the next best open-source model InternLM2-Math 20B at 37.7% (9.1 percentage point lead). The gap widens further against WizardMath-v1.1 7B (33.0%) and MetaMath 70B (26.6%).

**DeepSeekMath-Instruct approaches proprietary model performance on MATH.** Compared to closed-source models on MATH: GPT-4 scores 52.9%, Gemini Ultra scores 53.2%, Baichuan-3 scores 49.2%, GLM-4 scores 47.9%, and DeepSeekMath-Instruct scores 46.8%. The model is within 6.4 percentage points of the best proprietary model (Gemini Ultra), and surpasses Inflection-2 (34.8%), GPT-3.5 (34.1%), Gemini Pro (32.6%), and Grok-1 (23.9%). The paper's framing of "approaches the performance level of GPT-4" is reasonable given the 6.1 point gap, though "approaches" implies closer proximity than the raw numbers suggest.

**Tool-integrated reasoning shows strong performance, especially on MATH.** With tool use, DeepSeekMath-Instruct achieves 57.4% on MATH, surpassed among open-source models only by InternLM2-Math 20B (54.3%) — wait, the paper says DeepSeekMath-Instruct "approaches an accuracy of 60% on MATH, surpassing all existing open-source models," but Table 5 shows 57.4%, which is indeed higher than ToRA 34B (50.8%) and MAmmoTH 70B (41.8%). Compared to GPT-4 Code Interpreter's 69.7%, the gap is 12.3 percentage points — substantial but on a benchmark where tool use provides major advantages.

#### DeepSeekMath-RL 7B: Reinforcement Learning Improvements

Table 5 also presents DeepSeekMath-RL 7B results, showing the gains from GRPO training on DeepSeekMath-Instruct.

**RL provides consistent improvements across both in-domain and out-of-domain benchmarks.** On the in-domain task GSM8K (chain-of-thought), accuracy improves from 82.9% (Instruct) to 88.2% (RL), a gain of 5.3 percentage points. On MATH (chain-of-thought), the gain is 46.8% → 51.7%, an improvement of 4.9 percentage points. Crucially, these gains transfer to out-of-domain benchmarks: CMATH improves from 84.6% to 88.8% (4.2 points), MGSM-zh from 73.2% to 79.6% (6.4 points), and tool-integrated reasoning on GSM8K+Python from 83.7% to 86.7% (3.0 points). The paper emphasizes that "DeepSeekMath-RL 7B is only trained on chain-of-thought-format instruction tuning data of GSM8K and MATH, starting from DeepSeekMath-Instruct 7B," making the out-of-domain improvements notable evidence of generalization.

**DeepSeekMath-RL achieves 51.7% on MATH — the first open-source model to exceed 50%.** The paper presents this as a milestone: "obtaining an accuracy of over 50% on the competition-level MATH dataset for the first time within the open-source community." This claim is supported by the comprehensive baseline comparison in Table 5, where no other open-source model (including those up to 70B parameters) reaches 50% on MATH with chain-of-thought reasoning. The closest competitor is InternLM2-Math 20B at 37.7%, a 14-point gap.

**Comparison with closed-source models narrows.** DeepSeekMath-RL's 51.7% on MATH places it within 1.5 percentage points of GPT-4 (52.9%) and 1.5 points of Gemini Ultra (53.2%). The paper's claim of "approaching the performance level of Gemini-Ultra and GPT-4" is now stronger than for the Instruct model — at this level of performance, the model is genuinely competitive with the best proprietary systems on this specific benchmark.

**RL also improves tool-integrated reasoning, despite not being trained on tool-use data.** GSM8K+Python improves from 83.7% to 86.7% (3.0 points), MATH+Python from 57.4% to 58.8% (1.4 points). The gains are smaller than for chain-of-thought reasoning (reflecting the out-of-domain nature of tool-use data), but the fact that any improvement occurs at all indicates that the RL process enhances general reasoning capabilities that transfer across output formats.

#### Code Training and Mathematical Reasoning (Ablation)

Section 5.1.1 presents controlled experiments at the 1.3B scale to isolate the effect of code pre-training on mathematical reasoning. Tables 6 and 7 report the results.

**Two-stage code→math training achieves the best math-without-tool performance.** With code training followed by math training (Table 6), GSM8K reaches 21.9% vs. 20.5% for math-only training and 19.1% for general→math training. MATH reaches 15.3% vs. 13.1% and 14.4%, respectively. The absolute differences are modest (1-2 percentage points) but consistent.

**Code training dramatically improves program-aided mathematical reasoning.** GSM8K+Python jumps from 14.3% (general→math) to 17.4% (code→math), and MATH+Python from 6.7% to 9.4%. The code-only baseline (before any math training) already achieves 12.4% on GSM8K+Python and 10.0% on MATH+Python — remarkably, code pre-training *alone*, without any math-specific training, produces better Python-aided math performance than general pre-training followed by math training. This is strong evidence that programming skill is a key enabler of tool-based mathematical reasoning.

**One-stage mixed training preserves code capabilities but compromises pure math reasoning.** Table 7 shows that mixing code and math tokens in one stage largely avoids catastrophic forgetting: HumanEval achieves 29.3% (vs. 25.0% for code-only and 12.2% for code→math two-stage). However, math-without-tool performance suffers: GSM8K drops to 17.6% (vs. 21.9% for two-stage), MATH drops to 12.1% (vs. 15.3%). The paper conjectures a capacity limitation at 1.3B scale, which is plausible but unverified at 7B.

#### arXiv Paper Effectiveness (Negative Result)

Section 5.1.2 and Tables 8 and 9 evaluate whether arXiv papers improve mathematical reasoning. The results are consistently negative.

**arXiv training degrades or fails to improve performance across benchmarks and model scales.** On DeepSeek-LLM 1.3B (Table 8), MathPile training reduces GSM8K from 2.9% (no math training) to 2.7% and CMATH from 12.3% to 1.2%. ArXiv-RedPajama shows minimal improvements: GSM8K goes from 2.9% to 3.3%, but MMLU-STEM drops from 19.5% to 9.0%. On DeepSeek-Coder-Base-v1.5 7B, MathPile reduces GSM8K from 29.0% to 23.6% (a 5.4 point drop) and CMATH from 45.9% to 37.9%. ArXiv-RedPajama training is roughly neutral on most benchmarks but causes minor degradation on CMATH (45.9% → 42.6%).

**Formal theorem proving is also harmed by arXiv training.** Table 9 shows that on miniF2F-valid, MathPile reduces accuracy from 20.1% to 16.8%, and ArXiv-RedPajama causes an even larger drop to 14.8%. On miniF2F-test, the drops are from 21.7% to 16.4% (MathPile) and 11.9% (ArXiv-RedPajama). This is particularly counterintuitive given that arXiv papers contain formal mathematical content and theorem statements that would seem directly relevant to formal proof tasks.

#### Pass@K vs. Maj@K Analysis of RL Gains

Figure 7 presents the diagnostic analysis that separates RL's effects on fundamental capability vs. output distribution quality.

**Pass@K is essentially unchanged by RL.** On MATH at K=64, both Instruct and RL models achieve approximately 78% Pass@K (read from Figure 7). The curves for Pass@K-Instruct and Pass@K-RL nearly overlap across all values of K from 1 to 64. On GSM8K, Pass@K for both models approaches approximately 96% at K=64, with the curves again nearly indistinguishable. This is the critical negative result: RL does not enable the model to solve fundamentally new problems.

**Maj@K improves substantially, especially at higher K.** On MATH at K=64, Maj@K-RL reaches nearly 75%, compared to approximately 60% for Maj@K-Instruct — a gain of roughly 15 percentage points. The gap between the Maj@K curves widens as K increases: at K=1, the difference is small (roughly 45% vs. 40%), but by K=64, the RL model's majority voting accuracy has pulled far ahead. On GSM8K at K=64, Maj@K-RL approaches approximately 95%, vs. roughly 90% for Maj@K-Instruct.

**Interpretation: RL corrects the misalignment between generation probability and correctness.** The paper's analysis: "RL enhances Maj@K's performance but not Pass@K. These findings indicate that RL enhances the model's overall performance by rendering the output distribution more robust, in other words, it seems that the improvement is attributed to boosting the correct response from TopK rather than the enhancement of fundamental capabilities." The unchanged Pass@K confirms that the set of solvable problems is identical; the improved Maj@K confirms that within that set, correct solutions are more consistently ranked highly.

#### Iterative RL Performance

Figure 6 shows the effect of iterative GRPO with DeepSeekMath-Instruct 7B on GSM8K and MATH.

**Iterative RL provides meaningful additional gains.** On GSM8K, Iteration-0 (the Instruct baseline) starts at roughly 83-84% and ends at approximately 88% after 5,300 steps. Iteration-1 (first RL iteration) starts at roughly 84% and reaches approximately 88.5% — a small improvement over Iteration-0's endpoint. Iteration-2 starts at a similar level and reaches approximately 89%, providing roughly 1 additional percentage point beyond Iteration-1. On MATH, the pattern is clearer: Iteration-0 plateaus around 49%, Iteration-1 reaches roughly 50.5%, and Iteration-2 pushes to approximately 51.7%. The gains from iteration 1 to 2 are smaller than from 0 to 1, suggesting diminishing returns.

**The paper highlights iteration 1's outsized impact:** "we notice that the iterative RL significantly improves the performance, especially at the first iteration." This is consistent with the hypothesis that the initial reward model, trained on SFT-model outputs, becomes miscalibrated after the first round of policy updates, and retraining it on policy-generated data recovers some of the lost signal quality.

### Ablation Studies and Robustness Checks

**Online vs. offline sampling (RFT vs. Online RFT):** Figure 5 shows that Online RFT (which samples training data from the current policy model) significantly outperforms RFT (which uses stale samples from the initial SFT model) on both GSM8K and MATH. On MATH, the gap is approximately 1-2 percentage points throughout training, with Online RFT showing a steeper improvement curve in later steps. This confirms that data source freshness is a critical factor — training on the model's own current output distribution provides more relevant learning signal than training on the initial model's outputs. This finding is non-obvious because RFT is computationally cheaper (no online sampling needed), and the paper quantifies the performance cost of that shortcut.

**Outcome vs. process supervision (GRPO+OS vs. GRPO+PS):** Figure 5 shows that GRPO with process supervision (PS) outperforms GRPO with outcome supervision (OS) on both benchmarks. On MATH at 8,000 steps, GRPO+PS reaches roughly 30% accuracy vs. approximately 29% for GRPO+OS (read from Figure 5 for the 1.3B model). On GSM8K, the gap is approximately 1 percentage point (roughly 65.5% vs. 64.5%). The advantage is consistent but modest, suggesting that process supervision's finer-grained credit assignment provides marginal benefits for mathematical reasoning at this scale. The paper notes that "GRPO+PS shows superior performance compared to GRPO+OS, indicating the benefits of using fine-grained, step-aware gradient coefficients."

**Gradient coefficient type (Online RFT vs. GRPO):** Figure 5 directly compares Online RFT (binary gradient coefficient: 1 for correct, 0 for incorrect) against GRPO+OS (continuous gradient coefficient based on normalized reward). GRPO+OS consistently outperforms Online RFT by approximately 1-2 percentage points on both GSM8K and MATH. This isolates the effect of the gradient coefficient: both methods use online sampling (same data source) and rule/model-based reward, but GRPO's ability to provide differential reinforcement based on reward magnitude and to penalize incorrect outputs provides measurable benefits.

**Reward model training data (iterative vs. non-iterative RL):** Figure 6 shows that retraining the reward model between RL iterations provides additional gains. On MATH, Iteration-1 reaches approximately 50.5% vs. Iteration-0's ~49%, and Iteration-2 reaches ~51.7%. The diminishing returns suggest that the reward model becomes increasingly well-calibrated to the policy's output distribution, and further retraining yields smaller improvements.

**Effect of RL on out-of-domain benchmarks:** Table 5 demonstrates that RL trained exclusively on GSM8K and MATH chain-of-thought data improves performance on all evaluated benchmarks, including Chinese math (CMATH: 84.6% → 88.8%, MGSM-zh: 73.2% → 79.6%), tool-integrated reasoning (GSM8K+Python: 83.7% → 86.7%, MATH+Python: 57.4% → 58.8%), and Gaokao benchmarks (implied by the statement "it outperforms DeepSeekMath-Instruct 7B across all evaluation metrics"). This is a robustness check for the generalization of RL benefits: improvements are not restricted to the narrow training distribution.

**Model scale effect on arXiv results:** Tables 8 and 9 test arXiv effectiveness at two model scales (1.3B and 7B). The negative result replicates across scales, strengthening the conclusion that arXiv papers are genuinely ineffective for the evaluated benchmarks rather than being a small-model artifact.

**Corpus size vs. quality:** Figure 3 shows that DeepSeekMath Corpus outperforms Proof-Pile-2 at matched training budgets (e.g., at 50B tokens, corresponding to one Proof-Pile-2 epoch), confirming that the advantage is due to per-token quality rather than merely having more total data. This is a critical ablation because it rules out the alternative explanation that DeepSeekMath Corpus is simply larger.

### Critical Assessment

#### Does the paper demonstrate that web data contains sufficient high-quality mathematical content to train a competitive math model?

**Yes, with strong evidence.** The 1.3B-scale corpus comparison (Table 1) shows that DeepSeekMath Corpus-trained models substantially outperform those trained on all prior corpora across 8 diverse benchmarks. The learning curves in Figure 3 demonstrate both higher per-token quality (better performance at matched budgets) and better scaling (continued improvement beyond where other corpora plateau). The 7B-scale results (Table 2) show that a model trained primarily on web-sourced math data can outperform Minerva 540B on several benchmarks.

**Caveat:** The evaluation is limited to the specific benchmarks tested. There is no assessment of whether web-sourced math data teaches deep conceptual understanding vs. pattern-matching on problem formats similar to those in benchmarks. The improvement on out-of-domain Chinese benchmarks provides some evidence of genuine mathematical capability, but the scope of evaluation is still limited to multiple-choice and short-answer quantitative reasoning. The paper does not test on tasks requiring mathematical creativity, proof generation from scratch (beyond the limited miniF2F evaluation), or mathematical communication.

#### Does the paper demonstrate that arXiv papers are ineffective for mathematical reasoning?

**Yes, for the specific benchmarks and model scales tested.** The negative result is replicated across two model sizes (1.3B, 7B), two arXiv corpora (MathPile, ArXiv-RedPajama), and 10 benchmarks (8 mathematical + 2 formal theorem proving). No positive signal is observed anywhere.

**But the paper appropriately limits the scope of this claim.** Section 5.1.2 explicitly lists untested scenarios: arXiv might help on tasks like theorem informalization (not evaluated), when combined with other data types (only arXiv-only training is tested), or at larger model scales (beyond 7B). These are genuine limitations, not rhetorical hedges. A skeptical reader could argue that the paper overstates its case by saying "arXiv papers seem ineffective" rather than "arXiv-only training doesn't help on our specific benchmarks at our specific scales," but the paper's own caveats prevent this from being an overclaim.

**Missing experiment:** The paper includes 10% arXiv in the final DeepSeekMath-Base training mixture despite the negative result, suggesting the authors believe arXiv might have value in combination. But no ablation is run on the 7B model comparing the full mixture (with 10% arXiv) against a mixture where the 10% arXiv is replaced by additional web math or code data. This would directly test whether arXiv contributes positively when combined with other data types, which is the most policy-relevant question. Without this ablation, the paper's practice (including arXiv) is inconsistent with its findings (arXiv doesn't help).

#### Does the paper demonstrate that code training improves mathematical reasoning?

**Yes, with strong evidence at 1.3B scale.** The two-stage and one-stage experiments in Tables 6 and 7 consistently show that code pre-training benefits both tool-based and tool-free mathematical reasoning. The finding that code-only training (without any math data) achieves 12.4% on GSM8K+Python and 10.0% on MATH+Python is particularly compelling — it isolates code training's contribution to program-aided math.

**But the 7B model's architecture embeds this hypothesis rather than testing it.** DeepSeekMath-Base is initialized from a code model, and the training mixture includes 20% GitHub code. There is no 7B-scale ablation comparing code-initialized vs. general-model-initialized math training at matched compute. The paper therefore demonstrates that code+math training works at 7B scale, but does not quantify how much of the 7B model's advantage over Llemma or Mistral is due to code initialization vs. the DeepSeekMath Corpus vs. other factors. The 1.3B-scale results provide suggestive evidence, but scaling behavior is not guaranteed to be linear.

#### Does the paper demonstrate that GRPO is more efficient than PPO?

**The paper claims GRPO is more resource-efficient than PPO because it eliminates the value model, but provides no direct empirical comparison.** There is no head-to-head PPO vs. GRPO experiment at matched compute, matched memory, or matched wall-clock time. The paper's argument for efficiency is entirely architectural: the value model in PPO is "typically another model of comparable size as the policy model," which "brings a substantial memory and computational burden." This is a reasonable theoretical claim, but it is not empirically verified.

**The paper does not report GRPO's actual memory usage, training time, or FLOPs relative to PPO.** Without these numbers, a practitioner cannot determine whether GRPO's resource savings are 2× (as the architectural argument suggests), something more modest (if implementation optimizations for PPO exist), or something larger (if PPO's value model training requires additional overhead beyond memory). The paper's contribution on this point is conceptual rather than empirical — it proposes a more efficient architecture and shows that it works, but doesn't prove it is more efficient in practice.

**Additionally, the paper only evaluates GRPO on mathematical reasoning with outcome/process reward models.** Whether GRPO would work as well as PPO on other RLHF tasks (helpfulness, harmlessness, instruction following) where reward models may have different properties is untested. The claimed efficiency advantage is therefore domain-specific until demonstrated otherwise.

#### Does the paper demonstrate that RL improves mathematical reasoning by redistributing probability mass rather than expanding fundamental capability?

**Yes, with clean evidence from the Pass@K vs. Maj@K analysis.** Figure 7 provides a rare clear decomposition of RL's effect: Pass@K is unchanged (latent capability is identical), Maj@K is substantially improved (output distribution is more robust). This is a strong result because it provides a mechanistic explanation for *why* RL works, not just *that* it works.

**However, the analysis is limited to two benchmarks (GSM8K and MATH) and one temperature setting (0.7).** It is possible that at different sampling temperatures, or on harder out-of-domain benchmarks, RL might show some Pass@K improvement. The paper does not explore whether the unchanged Pass@K holds across different decoding strategies or problem difficulty levels. Additionally, the analysis only compares one Instruct model to one RL model — it does not show that this decomposition pattern holds across multiple RL runs, model scales, or training configurations. A more robust demonstration would show the same pattern in multiple settings.

#### Does the paper's unified RL paradigm genuinely advance understanding, or is it a post-hoc reframing?

**The unified paradigm (Equation 5) is a useful conceptual framework, but its novelty should not be overstated.** Expressing different training methods as gradient-based updates with varying coefficients is a standard technique in the RL and optimization literature. The paper's contribution is applying this lens systematically to the specific set of LLM post-training methods (SFT, RFT, DPO, PPO, GRPO) and extracting actionable insights about which axes matter most.

**The paradigm's empirical predictions are partially validated.** The paper predicts that Online RFT should outperform RFT (because of the data source axis), and this is confirmed in Figure 5. It predicts that GRPO should outperform Online RFT (because of the gradient coefficient axis), also confirmed. However, these predictions could have been made without the formal framework — they follow from general principles about the value of online learning and the limitations of binary reward signals. The framework's value is in making these comparisons precise and systematic, not in generating novel predictions that couldn't be arrived at through other reasoning.

**A genuine test of the framework's utility would be using it to predict the performance of a method not yet evaluated.** For example, the framework predicts that combining DPO's data source (offline sampling) with GRPO's gradient coefficient (continuous, group-normalized) should produce a specific performance profile. If the paper had tested such a hybrid and the framework's predictions held, that would provide stronger evidence of the framework's explanatory power. As it stands, the framework is primarily descriptive — it organizes known methods cleanly — rather than predictive.

#### Overall Strengths of the Experimental Design

- **Comprehensive benchmark coverage:** 10+ mathematical benchmarks spanning English, Chinese, multiple difficulty levels, and multiple reasoning formats (chain-of-thought, program-of-thought, tool-integrated, formal proof). This reduces the risk that results are benchmark-specific.

- **Multi-scale validation:** Corpus quality experiments at 1.3B scale before committing to 7B training, arXiv experiments at two scales, and RL experiments at both 1.3B (Figure 5) and 7B (Figure 6). This provides evidence that findings are not merely small-scale artifacts.

- **Clear negative results:** The arXiv experiments and the Pass@K vs. Maj@K decomposition are examples of well-designed diagnostics that produce clean, informative negative results. These are often more valuable than positive results because they constrain the space of plausible explanations.

- **Out-of-domain generalization testing:** RL is trained only on GSM8K and MATH chain-of-thought data, but evaluated on Chinese benchmarks, tool-integrated reasoning, and formal math. The consistent out-of-domain improvements (Table 5) strengthen the claim that RL provides genuine capability enhancement rather than benchmark overfitting.

#### Notable Weaknesses and Missing Experiments

- **No 7B-scale ablation of the data mixture:** The final DeepSeekMath-Base uses a specific mixture (56% DeepSeekMath Corpus, 4% AlgebraicStack, 10% arXiv, 20% GitHub code, 10% natural language). No ablation tests whether this specific ratio is optimal, whether arXiv could be removed without penalty, or whether more DeepSeekMath Corpus would be better. The 1.3B-scale corpus experiments test individual corpora in isolation, not mixtures.

- **No PPO vs. GRPO comparison:** The paper's headline claim about GRPO's efficiency is not empirically tested against PPO. A matched-budget comparison (same wall-clock time, same GPU memory) would directly quantify the efficiency gain.

- **Single model family, single pre-training framework:** All experiments use DeepSeek LLM architecture and training infrastructure. Whether the iterative data collection pipeline would produce similarly high-quality data for a differently architected model (e.g., Llama-family, Mistral-family) is unknown.

- **Limited formal math evaluation:** miniF2F with Isabelle is the only formal mathematics benchmark. The paper does not evaluate on other proof assistants (Lean, Coq, HOL Light) or on harder formalization benchmarks. The miniF2F results (24.6% test accuracy) are modest, and the paper doesn't analyze failure modes.

- **No confidence intervals or statistical testing:** All results are reported as point estimates. For a 500-question MATH test set, a 2 percentage point difference represents 10 questions — potentially within sampling noise. Without confidence intervals, it is unclear whether small differences (e.g., GRPO+PS vs. GRPO+OS in Figure 5) are statistically significant.

- **RL training data restriction is both a strength and a limitation:** Training RL only on GSM8K and MATH chain-of-thought data enables clean out-of-domain testing, but it also means the paper cannot determine how much RL would improve if trained on the full SFT data distribution. The out-of-domain improvements might be even larger with broader RL training data, or the model might overfit and lose generalization.

## 6. Limitations and Trade-offs

### The Hardest Problems Remain Unsolved — Test-Time Strategies Cannot Substitute for Fundamental Capability Gaps

**The assumption or constraint.** The paper's reinforcement learning approach operates on the assumption that the base model already possesses the latent capability to solve a problem — i.e., that at least one of K samples from the model has a non-zero probability of being correct (Pass@K > 0). GRPO can then redistribute probability mass to make those correct solutions more consistently ranked at the top. However, for problems where the base model's Pass@K is near zero, this mechanism provides no leverage. The paper explicitly acknowledges this in the context of the broader mathematical reasoning landscape:

> "DeepSeekMath is worse than GPT-4 on few-shot capability. GPT-4 could improve its performance with few-shot inputs, while DeepSeekMath shows similar performance in zero-shot and few-shot evaluation." (Section 6)

While this statement concerns few-shot generalization specifically, it reveals a broader capability ceiling: the 7B-parameter model, even after pre-training on 500B tokens and GRPO fine-tuning, lacks certain reasoning capacities that larger proprietary models possess — capacities that RL alone cannot create because RL only redistributes probability within the already-attainable solution space.

**The consequence.** For problem domains or difficulty levels where the base model's Pass@K is near zero, the entire RL pipeline provides essentially no benefit. This is most clearly visible in the formal theorem proving evaluation on miniF2F (Table 3), where DeepSeekMath-Base achieves only 25.8% on the validation set and 24.6% on the test set — meaning that for approximately 75% of formal proof problems, the model cannot generate a correct proof in 1 sample, and likely has near-zero Pass@K for those problems. GRPO training, which was applied only to GSM8K and MATH chain-of-thought data, cannot address this gap because it operates on problems where the model already occasionally succeeds. The paper does not evaluate DeepSeekMath-RL on miniF2F, but the Pass@K analysis in Section 5.2.2 (Figure 7) predicts that RL would not help on any problem where the base model's Pass@K is approximately zero.

More broadly, this limitation means that scaling model size (or radically improving pre-training data quality beyond what the DeepSeekMath Corpus achieves) remains necessary for tasks that are genuinely beyond the 7B model's reasoning frontier. The FLOPs-matched tradeoff literature (exemplified by the reference paper on test-time compute) provides a framework for deciding when inference compute can substitute for pretraining compute, but DeepSeekMath's finding that RL does not improve Pass@K establishes a sharp boundary: **no amount of post-training optimization can create capabilities that aren't already latent in the base model's output distribution**. For the hardest mathematical reasoning problems — and particularly for formal theorem proving — a 7B model simply may not have sufficient capacity, regardless of data quality or RL sophistication.

**What evidence exists in the paper.** The Pass@K vs. Maj@K analysis in Figure 7 (Section 5.2.2) provides direct evidence: Pass@K is essentially identical between the Instruct and RL models on both GSM8K and MATH. On MATH at K=64, both models achieve approximately 78% Pass@K. This means approximately 22% of MATH problems have Pass@64 ≈ 0 — the model cannot solve them even with 64 attempts. For these problems, RL provides no accuracy improvement (Maj@K cannot exceed Pass@K). The miniF2F results in Table 3 (25.8% valid, 24.6% test) suggest this fraction would be even larger for formal theorem proving. The paper does not evaluate DeepSeekMath-RL on miniF2F or on MATH difficulty sub-splits, so the precise fraction of problems where RL is ineffective is not quantified beyond the aggregate Pass@K curves.

**Mitigation status.** The paper acknowledges this limitation implicitly through the Pass@K analysis but does not propose solutions. Section 5.2.3 suggests exploring "out-of-distribution question prompts" as a future direction for RL data sources, which could potentially expose the model to a broader range of reasoning patterns during RL and expand the effective Pass@K frontier. However, this is speculative and was not tested. The fundamental constraint — that RL cannot create new reasoning capabilities — appears to be inherent to the approach and is not addressed by any proposed future work. Scaling model size or further improving pre-training data quality are the only paths forward for problems currently beyond the model's reach.

---

### Difficulty Estimation Cost Is Unaccounted For — The Iterative Data Pipeline Requires Substantial Human Annotation and Computational Overhead

**The assumption or constraint.** The iterative data collection pipeline (Section 2.1, Figure 2) relies on human annotators to label URL paths within identified math-related domains:

> "we manually annotate the URLs associated with mathematical content within these identified domains (e.g., mathoverflow.net/questions). Web pages linked to these URLs, yet uncollected, will be added to the seed corpus."

The paper provides no quantification of the human annotation effort required across the four iterations: how many domains were identified as math-related, how many URL paths were manually annotated, how many annotator-hours were required, what level of mathematical expertise was needed from annotators, or what quality control procedures were applied to ensure annotation consistency. The computational cost of the iterative pipeline is also not reported — running fastText inference over 40B HTML pages, training multiple classifier iterations, and performing domain-level analysis all consume non-trivial compute, but none of this cost is included in any reported budget or efficiency analysis.

**The consequence.** A practitioner attempting to replicate the pipeline for mathematics (or adapt it to another domain like law or medicine) cannot estimate the total resource requirements. The paper's narrative that "publicly accessible Common Crawl data contains valuable information" and that the pipeline is "meticulously engineered" is compelling, but without annotating the human and computational cost, the barrier to entry remains unclear. If four iterations required, say, 500 hours of expert mathematical annotator time plus significant compute for repeated fastText training and inference, the approach may only be feasible for well-resourced industrial labs, undermining the paper's implicit framing of this as a democratizing methodology.

Furthermore, the convergence criterion (stopping at the fourth iteration when 98% of data was already collected in the third) is domain-specific. A practitioner adapting the pipeline to a different domain has no guidance on how many iterations to expect, what convergence threshold to use, or how annotation effort should be allocated across iterations to maximize data quality improvement per unit of human effort. The paper treats the pipeline as a generalizable methodology ("applicable to other domains, such as coding") but provides no framework for estimating the cost of generalization.

**What evidence exists in the paper.** None. The paper does not report human annotation hours, annotator qualifications, annotation interface details, inter-annotator agreement metrics, or any other measure of the human cost of the pipeline. The paper does not report the computational resources consumed by the iterative fastText training and inference phases. The 500,000 positive and 500,000 negative examples used for fastText training are mentioned, but the total number of documents scored per iteration, the inference time, and the compute infrastructure are not described. The convergence statistic (98% overlap between iterations 3 and 4) is the only quantitative proxy for pipeline cost, and it only measures diminishing returns, not absolute effort.

**Mitigation status.** Not addressed. The paper does not suggest future work on reducing annotation cost, automating the URL path identification step, or developing heuristics for determining when sufficient data has been collected. The paper presents the pipeline as complete and successful but does not acknowledge the lack of cost transparency as a limitation. This is a significant gap for a paper whose primary contribution is a data collection methodology.

---

### Single Model Family, Single Pre-Training Framework — Generalization to Other Architectures and Training Paradigms Is Unknown

**The assumption or constraint.** All experiments use the DeepSeek LLM architecture and training framework (DeepSeek-AI, 2024). The 7B model architecture, the HAI-LLM training framework, the specific tokenizer (vocabulary size 100K), and the DeepSeek LLM training recipe (multi-step learning rate schedule with specific decay factors, AdamW with β₁=0.9, β₂=0.95, weight_decay=0.1) are held constant throughout. The paper does not experiment with any other model family (e.g., Llama, Mistral, Qwen) or training framework. The paper states:

> "we believe this model is representative of the capabilities of many contemporary LLMs" (Section 4, paraphrased)

This claim of representativeness is not empirically supported.

**The consequence.** Three specific uncertainties arise from this limitation:

First, the DeepSeekMath Corpus's effectiveness may interact with model architecture in ways that don't generalize. For instance, the DeepSeek LLM architecture may be particularly well-suited to learning from mathematical web text (perhaps due to its attention patterns, positional encoding scheme, or vocabulary coverage of mathematical symbols), while a Llama-architecture model trained on the same corpus might not achieve the same performance. The paper provides no evidence either way.

Second, the finding that code pre-training benefits mathematical reasoning (Section 5.1.1) was demonstrated only on DeepSeek-LLM 1.3B. The magnitude of this benefit may depend on the specific code pre-training data, the architecture's capacity for cross-task transfer, and the interaction between code and math token representations — all of which could vary across model families. A practitioner using a Mistral or Llama base model cannot assume the same benefit will materialize.

Third, the GRPO algorithm's performance relative to PPO is evaluated only within the DeepSeek ecosystem. The paper's theoretical argument for GRPO's efficiency (eliminating the value model) is architecture-agnostic, but the practical performance — training stability, sensitivity to hyperparameters, interaction with the KL penalty estimator — may depend on implementation details of the training framework. The paper does not release GRPO training code at the time of writing for independent verification.

**What evidence exists in the paper.** The paper provides no cross-architecture experiments. The 1.3B-scale corpus quality experiments (Table 1) use DeepSeek-LLM 1.3B; the 7B-scale experiments use DeepSeek-Coder-Base-v1.5 7B; the RL experiments use DeepSeekMath-Instruct 7B. All models share the same architecture family. The paper's comparisons to external models (Mistral 7B, Llemma 34B, Minerva 540B) benchmark DeepSeekMath against different architectures, but this only demonstrates that DeepSeekMath outperforms those models — it does not demonstrate that the DeepSeekMath Corpus would be equally effective if used to train those other architectures.

**Mitigation status.** The paper does not acknowledge this as a limitation. The claim of representativeness is presented without caveat. The release of the DeepSeekMath Corpus (if made publicly available) would partially mitigate this limitation by enabling other researchers to test the corpus on different model architectures, but the paper does not commit to releasing the corpus, and as of publication, independent verification is not possible. The GRPO algorithm is described in sufficient detail for reimplementation, but the absence of cross-framework validation means that practitioners adopting GRPO for non-DeepSeek models are operating without empirical guidance on expected performance.

---

### Single Benchmark Domain — Mathematical Reasoning Results May Not Transfer to Other Reasoning Tasks

**The assumption or constraint.** All RL experiments and the vast majority of pre-training evaluations are conducted on mathematical reasoning benchmarks. While the paper evaluates natural language understanding (MMLU), general reasoning (BBH), and code generation (HumanEval, MBPP) for the base model (Table 4), the RL stage is trained exclusively on ~144K chain-of-thought questions from GSM8K and MATH and only evaluated on mathematical benchmarks (Table 5, Figure 5, Figure 6). The paper does not report the effect of GRPO on MMLU, BBH, HumanEval, or MBPP for the DeepSeekMath-RL model.

Additionally, the DeepSeekMath Corpus was constructed using a classifier trained specifically to recognize mathematical web pages. While the paper claims the iterative pipeline is "also applicable to other domains, such as coding," no experiments validate this claim by constructing a code-specific corpus and evaluating on code benchmarks. The entire data quality validation (Section 2.2, Table 1, Figure 3) uses only mathematical benchmarks.

**The consequence.** A practitioner interested in applying GRPO to non-mathematical reasoning tasks (e.g., logical reasoning, scientific QA, legal reasoning, medical diagnosis) has no empirical evidence about whether the algorithm's benefits transfer. The Pass@K vs. Maj@K decomposition (Figure 7) provides a mechanistic explanation for GRPO's effectiveness — it corrects misalignment between generation probability and correctness — but this mechanism may operate differently in domains where:
- Correctness is more subjective or multi-dimensional (e.g., essay quality, dialogue helpfulness)
- The reward model is harder to train because ground-truth correctness labels are unavailable
- The base model's Pass@K is lower, meaning there is less latent capability for RL to surface

The paper's unified paradigm (Section 5.2.1, Equation 5) is domain-agnostic, but the empirical validation is entirely mathematical. The finding that RL improves Maj@K but not Pass@K might be specific to mathematical reasoning, where correctness is binary and well-defined. In open-ended generation tasks, the distinction between "capability" and "output distribution quality" may blur or require different metrics to operationalize.

Similarly, the iterative data collection pipeline's success on mathematical web data does not guarantee success on other domains. Mathematics has distinctive surface features (LaTeX notation, equation formatting, specialized symbols) that make it relatively easy for fastText classifiers to distinguish from general web content. A domain like law or medicine may have subtler linguistic markers, making classifier-based recall less effective. The convergence behavior (four iterations to reach 98% recall) is likely domain-specific, but the paper provides no framework for predicting how many iterations a new domain would require.

**What evidence exists in the paper.** Table 5 shows that RL trained on English math data generalizes to Chinese math benchmarks (CMATH: 84.6% → 88.8%, MGSM-zh: 73.2% → 79.6%) and to tool-integrated math reasoning (GSM8K+Python: 83.7% → 86.7%, MATH+Python: 57.4% → 58.8%). This provides evidence of cross-lingual and cross-format generalization within mathematics, but not cross-domain generalization. No non-math benchmark results are reported for the RL model. The base model's MMLU and BBH improvements from math training (Table 4: MMLU 49.1% → 54.9%, BBH 55.2% → 59.5%) suggest that math pre-training transfers to general reasoning, but whether GRPO would further improve these scores (or degrade them through over-optimization for math-specific patterns) is unknown.

**Mitigation status.** Partially acknowledged. The paper's RL training data restriction (using only GSM8K and MATH chain-of-thought data, approximately 144K of the 776K SFT examples) is framed as a deliberate choice to "investigate the impact of RL on benchmarks that lack data throughout the RL phase" (Section 4.2). This design tests cross-benchmark generalization within mathematics but not cross-domain generalization. The paper does not discuss whether GRPO would be expected to help or harm performance on non-math tasks, and does not suggest experiments to test this. Section 5.2.3 identifies "how to enhance the generalization ability of the reward model" as a future direction, acknowledging that reward model generalization to out-of-distribution questions is a challenge, but this discussion is limited to mathematical question distributions, not entirely different domains.

---

### No FLOPs-Matched or Wall-Clock Comparison Between Pre-Training and Post-Training Compute Allocation

**The assumption or constraint.** The paper presents DeepSeekMath as the result of a sequential pipeline: pre-training on 500B tokens, supervised fine-tuning on 776K examples, and GRPO reinforcement learning on 144K questions. Each stage consumes substantial compute, but the paper provides no analysis of how the total compute budget could be optimally allocated across these stages. For instance, would training on 600B tokens of pre-training data with no RL outperform the current 500B + SFT + RL pipeline? Would a larger SFT dataset (e.g., using all 776K examples for RL instead of only 144K) improve RL performance further, or does the RL benefit saturate with training data quantity?

The paper also makes no comparison between its approach and alternative resource allocation strategies. A practitioner with a fixed compute budget must decide how to divide it between:
- Collecting more / better pre-training data (additional Common Crawl iterations)
- Training on more tokens during pre-training
- Expanding the SFT dataset
- Running more iterations of GRPO
- Scaling to a larger model architecture

The paper provides no guidance on these tradeoffs because each stage's contribution is evaluated in isolation — the DeepSeekMath Corpus is validated at 1.3B scale, the 7B model is trained with a fixed data mixture for 500B tokens, SFT is applied for 500 steps, and RL is run for 2 iterations — without comparing alternative budget splits.

**The consequence.** A practitioner cannot determine whether DeepSeekMath's strong performance is primarily attributable to the DeepSeekMath Corpus (pre-training data quality), the code initialization (model architecture choice), the SFT data composition (instruction tuning), or the GRPO algorithm (RL). Each of these contributes, but their relative importance is unknown. If a team has limited resources and can only invest in one of these components, the paper provides no guidance on which investment yields the highest marginal return.

Furthermore, the paper cannot claim that GRPO is "efficient" in any absolute sense beyond the architectural argument about removing the value model. Efficient relative to what? If GRPO requires 2× fewer GPU-hours than PPO for the RL stage but the RL stage itself provides only a 4.9 percentage point improvement on MATH (46.8% → 51.7%), and that same compute could have been spent on pre-training additional tokens that might yield a larger improvement, then GRPO's efficiency advantage over PPO is irrelevant — the entire RL stage might be an inefficient use of compute compared to more pre-training. The paper does not perform this analysis.

**What evidence exists in the paper.** The paper reports the performance of each stage independently: DeepSeekMath-Base (pre-training only), DeepSeekMath-Instruct (pre-training + SFT), and DeepSeekMath-RL (pre-training + SFT + RL). The incremental contributions can be approximately estimated: on MATH, Base achieves 36.2% (Table 2), Instruct achieves 46.8% (Table 5, a 10.6 point gain from SFT), and RL achieves 51.7% (a 4.9 point gain from GRPO). However, these numbers are not compute-matched — SFT and RL use vastly less compute than pre-training, so the larger absolute gain from SFT does not imply it is more compute-efficient. The paper provides no FLOPs accounting for any stage.

The iterative RL experiments (Figure 6) show that a second RL iteration provides diminishing returns (MATH: ~49% → ~50.5% → ~51.7%), but this is only compared within the RL stage, not against alternative uses of the same compute. The 1.3B-scale corpus experiments (Table 1) compare pre-training on different corpora at matched token counts (150B tokens each), but do not compare pre-training against SFT or RL at matched compute.

**Mitigation status.** Not addressed. The paper does not discuss compute allocation tradeoffs, does not report FLOPs or GPU-hours for any training stage, and does not suggest future work on compute-optimal allocation across pre-training, SFT, and RL. This is a significant omission for a paper whose primary practical contribution is a recipe for building math-capable models — a recipe without cost estimates or allocation guidance is incomplete.

---

### Reward Model Quality and Over-Optimization Behavior Are Not Characterized

**The assumption or constraint.** GRPO's effectiveness depends entirely on the reward model's ability to accurately score the relative quality of sampled outputs. If the reward model is noisy, poorly calibrated, or exploitable, GRPO's gradient coefficient (Equation 21) may reinforce undesirable behaviors or fail to penalize errors effectively. The paper acknowledges this dependency in Section 5.2.3:

> "it is impossible to ensure the reward signal is always reliable, especially in extremely complex tasks. For example, even the PRM800K datasets, which have been carefully annotated by well-trained annotators, still contain approximately 20% of incorrect annotations."

However, the paper provides no characterization of its own reward model's quality: no evaluation of reward model accuracy on held-out data, no analysis of reward model calibration (do higher scores correspond monotonically to higher actual correctness rates?), no measurement of reward model agreement with human judgments, and no investigation of whether GRPO causes the policy to exploit reward model weaknesses (reward hacking).

**The consequence.** Without understanding the reward model's error characteristics, a practitioner cannot predict when GRPO will help vs. harm. Two specific failure modes are plausible given the paper's design but are not evaluated:

First, **reward model distribution shift**: the initial reward model is trained on outputs from the SFT model. As GRPO updates the policy, the policy's output distribution shifts, and the reward model may become increasingly miscalibrated on the new distribution. The iterative RL procedure (Algorithm 1) retrains the reward model on policy-generated data to address this, but the paper does not measure how much the reward model's accuracy degrades between iterations or how effectively retraining restores it. If the reward model's accuracy drops substantially during an iteration, the GRPO updates in the latter half of that iteration may be based on noisy or misleading reward signals, potentially degrading the policy.

Second, **reward hacking through stylistic rather than substantive improvements**: GRPO may learn to produce outputs that score highly under the reward model without actually being correct more often. For example, the policy might learn to format answers in a particular way that the reward model associates with correctness, or to include certain key phrases that trigger high reward scores. The paper's finding that Pass@K is unchanged (Figure 7) provides some reassurance — if reward hacking were occurring, we might expect Pass@K to degrade as the policy exploits reward model weaknesses at the expense of actual solution quality. However, this is indirect evidence; direct measurement of reward model accuracy on policy-generated outputs across training would be more conclusive.

**What evidence exists in the paper.** Very little. The paper states that the reward model is trained following Wang et al. (2023b) and is initialized from DeepSeekMath-Base 7B with a learning rate of 2e-5, but provides no evaluation of reward model performance. The process supervision variant (GRPO+PS) outperforms outcome supervision (GRPO+OS) in Figure 5, which the paper interprets as evidence that process-level rewards are more informative. However, this does not reveal whether either reward model is accurate in absolute terms — process supervision could be merely less noisy, not actually correct on a per-step basis.

The iterative RL results (Figure 6) show that retraining the reward model helps, which is consistent with (but does not prove) the presence of reward model distribution shift. The performance gains from iteration 1 to iteration 2 are smaller than from iteration 0 to iteration 1, which could indicate that the reward model becomes better-calibrated and further retraining provides diminishing returns, or could simply reflect the policy approaching a performance plateau for other reasons.

**Mitigation status.** Partially acknowledged as future work. Section 5.2.3 identifies three directions for reward model improvement: enhancing generalization ability, reflecting uncertainty, and building high-quality process reward models. However, these are presented as aspirational future directions, not as mitigations for limitations of the current approach. The paper does not conduct any experiments on reward model robustness, calibration, or exploitation resistance. The "WEAK-TO-STRONG alignment" framing (Burns et al., 2023) is invoked as a long-term aspiration, but no weak-to-strong experiments are conducted. A practitioner deploying GRPO with a reward model of unknown quality has no guidance on what level of reward model accuracy is sufficient for GRPO to be beneficial rather than harmful.

## 7. Implications and Future Directions
- Field impact
  - A 7B open model reaching >50% on MATH (Table 5) resets expectations for the parameter count required for competitive math reasoning. The work also establishes scalable web data mining and critic‑free RL as practical tools for domain specialization.

- Research enabled/suggested (Section 5.2.3, Section 6)
  - Data
    - Extend the mining pipeline to other domains (e.g., scientific reasoning, formal methods) and languages; refine coverage for geometry and theorem proving.
    - Explore hybrid corpora where arXiv complements curated web/math code with better integration.
  - Algorithms
    - Robust RL under noisy rewards; integrate uncertainty estimates from reward models.
    - Explore stronger online sampling/decoding (e.g., tree-of-thoughts) and efficient inference for exploration (speculative decoding, memory-optimized serving).
    - Iterative co-training of policy and reward models at scale; step-aware (process) rewards learned from cheaper supervision.
  - Evaluations and applications
    - Broaden to multi-modal math (diagrams), symbolic reasoning with tool stacks, and real classroom/tutoring settings.
    - Use GRPO-style alignment for other structured domains (law, finance) where relative judgments and process feedback are available.

> Headline result: Table 5 shows `DeepSeekMath-RL 7B` reaches 51.7% Top‑1 on MATH with chain‑of‑thought reasoning and 58.8% with tool‑integrated reasoning, beating all open models from 7B to 70B and most closed models except GPT‑4 and Gemini Ultra.

> Data quality and scale: Figure 3 and Table 1 demonstrate that models trained on the 120B‑token DeepSeekMath corpus learn faster and longer than those trained on OpenWebMath (13.6B) or Proof‑Pile‑2 (51.9B), with large gains on both English and Chinese benchmarks.

> Methodological advance: Figure 4 and Equations (3)–(4) show how GRPO removes the critic by using group-relative normalized rewards and a direct KL penalty, reducing memory while improving over Online RFT (Figure 5) and benefiting further from iterative updates (Figure 6).
