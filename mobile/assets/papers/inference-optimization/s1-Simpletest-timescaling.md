# s1: Simple test-time scaling

**ArXiv:** [2501.19393](https://arxiv.org/abs/2501.19393)

## 🎯 Pitch

This paper introduces a remarkably simple and transparent recipe for test-time scaling in reasoning language models: just fine-tune an existing model on 1,000 expertly-curated chain-of-thought examples and control reasoning depth at inference using a decoding intervention called 'budget forcing.' This approach yields a 32B-parameter open model (s1-32B) that not only demonstrates true, monotonic accuracy gains as more compute is spent per question, but also surpasses leading closed models on math benchmarks with orders of magnitude less supervision. The simplicity, sample efficiency, and full openness of this method democratize advanced reasoning research and set a new standard for reproducible, controllable interpretability in LLM test-time computation.

---

## 1. Executive Summary

This paper introduces **budget forcing**, a simple test-time intervention that controls reasoning duration, and demonstrates that supervised fine-tuning on only 1,000 carefully curated reasoning samples produces strong test-time scaling behavior. Using Qwen2.5-32B-Instruct fine-tuned on the s1K dataset — 1,000 questions selected for difficulty, diversity, and quality with reasoning traces distilled from Gemini Flash Thinking — the resulting s1-32B model matches or exceeds o1-preview on MATH500 and AIME24 while being the most sample-efficient open-data reasoning model. Budget forcing operates by suppressing the end-of-thinking token delimiter and appending “Wait” to force continued reasoning (sequential scaling, extending AIME24 accuracy from 50% to 57%), or by forcibly inserting the delimiter to cap thinking tokens, giving the method perfect controllability over test-time compute. Ablations establish that combining difficulty, diversity, and quality criteria is essential — random, diverse-only, and longest-only selections each degrade performance by roughly 30% on AIME24 on average — and that budget forcing’s controllability and positive scaling slope outperform token-conditional, step-conditional, class-conditional, and rejection sampling methods. The approach extrapolates performance with additional test-time compute only until repetitive loops emerge — beyond roughly six forced “Wait” interventions, scaling flattens — establishing that simple sequential prompting-based scaling works for moderate compute expansion but encounters a ceiling that parallel methods such as REBASE tree search can complement.

## 2. Context and Motivation

### The Core Problem: Reproducing Test-Time Scaling in a Simple, Open, and Sample-Efficient Way

This paper addresses a specific, concrete gap in the rapidly evolving landscape of language model reasoning. In September 2024, OpenAI released the o1 model series, which demonstrated a new capability: **test-time scaling** — the model's performance improved predictably as it was allowed to spend more computation at inference time generating "reasoning traces" (chain-of-thought steps) before producing its final answer. This was a significant departure from previous scaling paradigms that focused almost exclusively on scaling *training* compute. OpenAI demonstrated consistent accuracy gains as they increased inference-time computation, producing what the community began calling "scaling curves" for test-time compute.

However, OpenAI did not publicly share their methodology. The o1 technical report was deliberately vague, mentioning only that the models were trained using "large-scale reinforcement learning" and could "think before they respond." This opacity created an urgent gap in the open research community: **nobody outside OpenAI knew how to build a model that exhibits test-time scaling behavior, and nobody had openly replicated the characteristic scaling curves.**

This gap had immediate practical consequences. Several concurrent efforts attempted to replicate o1 — including DeepSeek R1 (DeepSeek-AI et al., 2025), which successfully achieved o1-level performance but required massive computational resources (reportedly ≫800K training examples with multiple stages of reinforcement learning), and Kimi k1.5 (Team et al., 2025), which also employed large-scale RL. Other approaches explored Monte Carlo Tree Search (Gao et al., 2024b), multi-agent debate (Qin et al., 2024), and various distillation pipelines (Huang et al., 2024b). But as the authors state in their introduction:

> "despite the large number of o1 replication attempts, none have openly replicated a clear test-time scaling behavior."

This is the precise gap the paper aims to fill. The question they pose is explicit and ambitious: **what is the simplest approach to achieve both test-time scaling and strong reasoning performance?** The word "simplest" is crucial — the paper is not trying to beat o1 at all costs, but rather to find the minimal recipe that produces the core phenomenon of test-time scaling in an open, reproducible way.

### Why This Problem Matters

The importance of this problem extends beyond academic curiosity about replicating a proprietary system. There are several dimensions to its significance:

**1. Understanding the mechanism behind test-time scaling.** Before this work, it was unclear whether test-time scaling required complex reinforcement learning training (which would imply that the model needs to learn to *value* its own intermediate reasoning steps through reward signals), or whether simpler supervised fine-tuning on reasoning traces could suffice. If the latter were true — and this paper shows it is — it would suggest that the capacity for reasoning is already latent in pretrained models (from their exposure to trillions of tokens of text during pretraining) and merely needs to be "activated" through targeted fine-tuning. The authors explicitly reference this hypothesis through the lens of the "Superficial Alignment Hypothesis" from LIMA (Zhou et al., 2023):

> "We hypothesize that the model is already exposed to large amounts of reasoning data during pretraining which spans trillions of tokens. Thus, the ability to perform reasoning is already present in our model. Our sample-efficient finetuning stage just activates it and we scale it further at test time with budget forcing."

If this hypothesis is correct, it fundamentally changes how researchers should think about building reasoning models: the bottleneck is not capability acquisition (which requires massive RL training) but capability *elicitation* (which can be achieved through sample-efficient SFT and inference-time interventions).

**2. Democratizing reasoning research.** The compute requirements for DeepSeek R1 and Kimi k1.5 — requiring millions of training samples, multiple training stages, and enormous GPU clusters — place open reasoning research out of reach for most academic labs. If the same scaling behavior can be achieved with 1,000 training examples and 26 minutes of training on 16 H100 GPUs (as this paper does), it dramatically lowers the barrier to entry. This enables broader experimentation with reasoning architectures, test-time interventions, and scaling strategies.

**3. Sample efficiency as a scientific signal.** The extreme sample efficiency achieved — 1,000 examples versus ≫800K for r1-distill — isn't just a practical convenience; it's a scientific finding. It suggests that the data selection *quality* matters far more than quantity for reasoning fine-tuning, and that the community's default approach of "scrape everything and throw it at the model" may be deeply suboptimal for reasoning tasks. This connects to a broader theme in the instruction tuning literature (Zhou et al., 2023) but extends it to the much more challenging domain of multi-step mathematical and scientific reasoning.

**4. Open infrastructure for test-time methods.** By releasing the model, data, and code under an open-source license, the paper provides a shared platform for testing future test-time compute methods. Prior to this, researchers studying test-time scaling methods had to work with proprietary models (o1) or models that didn't exhibit clean scaling behavior. s1-32B provides a controllable, reproducible testbed where the amount of test-time computation can be precisely manipulated via budget forcing.

### Where Prior Approaches Fall Short

The paper identifies specific limitations in prior and concurrent work that motivate its approach.

**Large-scale RL approaches are effective but opaque and expensive.** DeepSeek R1 (DeepSeek-AI et al., 2025) demonstrated o1-level performance through reinforcement learning at massive scale — millions of training examples, multiple stages of RL and SFT. While this showed that open models *can* achieve test-time scaling through RL, it didn't answer whether RL is *necessary*. The enormous compute requirements and multi-stage training pipeline make it difficult to isolate which components are essential and which are incidental. Similarly, Kimi k1.5 (Team et al., 2025) used large-scale RL with careful reward engineering, but their methodology is similarly resource-intensive. As the authors note:

> "DeepSeek R1 has successfully replicated o1-level performance, also employing reinforcement learning via millions of samples and multiple training stages."

The crucial unasked question is: could the same or similar behavior be achieved through far simpler means?

**Distillation-based approaches lack test-time scaling.** Models like Sky-T1 (Team, 2025) and Bespoke-Stratos (Labs, 2025) trained on tens of thousands of distilled reasoning examples from QwQ-32B or R1 and achieved strong static performance. However, these models did not demonstrate the characteristic *scaling* behavior — the ability to improve predictably with additional test-time compute. They were strong reasoners but not test-time scalable reasoners. The difference is important: a model that scales at test time can adapt its computation to problem difficulty (spending more time on harder problems), while a static model has a fixed cost-performance tradeoff. As Figure 4(b) in the paper shows, even majority voting with the base Qwen2.5-32B-Instruct model does not catch up to s1-32B with sequential scaling — the scaling curve has a fundamentally different (steeper) slope.

**Prior test-time methods lack controllability or degrade performance.** The paper identifies a family of methods for controlling test-time compute that all have significant limitations:

- **Rejection sampling** (sampling until generation fits a length constraint) shows *inverse* scaling — longer generations tend to be *less* accurate than shorter ones, because shorter generations correlate with being on the right track from the start, while longer ones often involve backtracking and confusion. The paper demonstrates this clearly in Figure 6, where accuracy *decreases* as the allowable thinking token budget increases from 3,072 to 5,120 tokens. This is a counterintuitive and important negative result: simply allowing the model to "think longer" doesn't help if longer thinking is correlated with confusion.

- **Token-conditional control** (specifying an upper bound on thinking tokens in the prompt) fails because current models cannot reliably count tokens. The paper shows (Table 12) that the model generates similar numbers of tokens regardless of the instructed limit, often overshooting. This is consistent with independent findings that even o1-mini cannot follow token-length instructions (Zhang & Chen, 2024).

- **Step-conditional control** (specifying an upper bound on thinking steps) suffers from compensation effects: the model makes each step longer when given fewer steps, and vice versa, keeping total tokens roughly constant (Table 13). The model "learns to hack its way around the compute constraint," as the authors observe, leading to inadequate controllability.

- **Class-conditional control** (vague prompts asking the model to "think longer" or "think shorter") provides some separation effect but is imprecise and doesn't reliably improve performance (Table 14). Long-thinking prompts produce longer traces but not consistently better accuracy.

**Tree search methods require additional reward models and compute.** REBASE (Wu et al., 2024b), a state-of-the-art tree search method using process reward models, can complement sequential scaling (Figure 7) but adds significant overhead: each step requires an additional forward pass through the reward model. Moreover, PRM training adds another layer of complexity to the pipeline. The paper's budget forcing requires no auxiliary models — just the base reasoning model and a simple token manipulation strategy.

**No open method provided both clean scaling curves and strong performance simultaneously.** Before this paper, the landscape was split: RL-based methods (R1) achieved strong performance but with massive compute costs; distillation methods (Sky-T1, Bespoke) achieved moderate performance but without scaling behavior; and test-time methods for existing models (majority voting, rejection sampling) provided scaling but from a lower-performance base. The paper's contribution is finding a point in the design space — SFT on 1K carefully selected examples + simple budget forcing — that simultaneously provides strong static performance and clean scaling curves, all with minimal resources.

### How This Paper Positions Itself

The paper positions itself explicitly as a search for **simplicity** and **sample efficiency** in reasoning model development. This positioning is evident in several key framing choices:

**The title — "s1: Simple test-time scaling."** The simplicity claim is central and deliberate. The paper is not claiming to beat o1 on raw performance (it doesn't — o1 achieves 74.4% on AIME24 vs. s1-32B's 56.7%, though it exceeds o1-preview's 44.6%). Instead, it claims to achieve test-time scaling behavior through simpler means than any prior open work, while remaining competitive with o1-preview.

**The explicit focus on minimal data.** The paper draws a direct lineage to LIMA (Zhou et al., 2023), which showed that 1,000 carefully chosen examples could suffice for instruction alignment. The authors extend this insight to reasoning:

> "This is similar to the 'Superficial Alignment Hypothesis' presented in LIMA, where the authors find that 1,000 examples can be sufficient to align a model to adhere to user preferences."

By positioning their work as an extension of LIMA to reasoning, the authors frame their finding as supporting the broader hypothesis that pretraining already encodes substantial reasoning capability, and fine-tuning serves mainly to format and elicit it.

**Three guiding data principles — difficulty, diversity, quality — that are systematically ablated.** Rather than making vague claims about data curation, the paper formalizes three criteria and tests each through ablation (Table 2). The ablation structure (1K-random, 1K-diverse, 1K-longest, 59K-full) is designed to demonstrate that *all three criteria jointly* are necessary — no single criterion or simple combination achieves comparable performance. This is important because it provides a recipe that others can follow, rather than leaving data selection as a mysterious art.

**A formal evaluation framework for test-time scaling methods.** Section 3.2 introduces three quantitative metrics — Control, Scaling, and Performance — for evaluating test-time scaling methods, and applies them systematically to compare budget forcing against baselines (Table 3). This formalization moves the discussion from qualitative "this method seems to work" to quantitative comparison, enabling future researchers to evaluate new methods against a clear standard. Budget forcing achieves 100% Control (perfect adherence to compute limits), positive Scaling (15, the highest among methods with ≥ 50% Control), and the best Performance (56.7% on AIME24 using the method). This three-metric framework implicitly argues that prior methods failed on at least one dimension: conditional control methods had low Control; rejection sampling had negative Scaling; budget forcing dominates on all three.

**Connection to parallel scaling as a complement, not a competitor.** The paper doesn't claim that sequential budget forcing is the final answer to test-time scaling. Instead, Section 6.2 acknowledges its limitations (flattening at ~6 interventions, context window constraints) and discusses how parallel methods like REBASE and majority voting can complement sequential scaling for further gains (Figure 7). This positions budget forcing as one tool in a broader toolkit, not as a universal solution. The paper's contribution is establishing that simple sequential scaling works and is complementary to parallel scaling, rather than claiming it supersedes all alternatives.

**Openness as a scientific value.** The paper concludes with an explicit impact statement about transparency:

> "Recent advances in reasoning, such as OpenAI's o1 and DeepSeek's r1, lack transparency, limiting broader research progress. Our work aims to push the frontier of reasoning in a fully open manner."

This framing positions the paper not just as a technical contribution but as a response to the increasingly closed nature of frontier reasoning research, with the explicit goal of democratizing access to test-time scaling methodology.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper builds a **two-component reasoning system** consisting of (1) a language model fine-tuned on a small, carefully curated dataset of reasoning traces, and (2) a test-time intervention called **budget forcing** that controls how long the model spends "thinking" before answering. The core problem it solves is: **how can we achieve test-time scaling behavior — predictable performance improvement with additional inference compute — through the simplest possible means?** The shape of the solution is remarkably minimal: supervised fine-tuning on 1,000 examples teaches the model to produce long-form reasoning traces, and a simple token-level manipulation at test time (suppressing or inserting a delimiter token) controls the duration of that reasoning with perfect precision, enabling the characteristic scaling curves that previously required massive reinforcement learning pipelines.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has two major components that operate sequentially:

1. **Data Curation Pipeline (s1K):** Takes 59,029 raw questions from 16 sources, distills reasoning traces from Gemini Flash Thinking API, and applies three-stage filtering (quality → difficulty → diversity) to select exactly 1,000 training examples. This component produces the supervised fine-tuning dataset.

2. **Training + Inference System (s1-32B + Budget Forcing):** Takes Qwen2.5-32B-Instruct (a pretrained and instruction-tuned 32B parameter model), fine-tunes it on s1K for 26 minutes on 16 H100 GPUs, producing s1-32B. At inference time, budget forcing manipulates the model's generation: to *cap* thinking, it forcibly appends `<|im_start|>answer` which acts as an end-of-thinking delimiter, making the model transition to answer mode; to *extend* thinking, it suppresses the model's generation of this delimiter and instead appends the string `"Wait"`, causing the model to continue reasoning and often self-correct.

Information flows as follows: a question enters → if using budget forcing to cap thinking, the system generates until a token limit is reached, then forces the delimiter and extracts the answer; if using budget forcing to extend thinking, the system monitors for the end-of-thinking delimiter during generation, suppresses it when detected, appends `"Wait"`, and lets generation continue — repeating this suppression up to 6 times — then extracts the best answer from the full reasoning chain.

### 3.3 Roadmap for the Deep Dive

- **First, the data curation pipeline (§2 in the paper):** How 59K raw samples are distilled from Gemini, and the three-stage filtering process (Quality, Difficulty, Diversity) that selects exactly 1,000 training examples. This ordering is essential because the quality of s1K determines everything downstream — the ablations show that alternative selection strategies degrade performance by ~30%.

- **Second, supervised fine-tuning (SFT):** How Qwen2.5-32B-Instruct is trained on s1K, including the token delimiter format that separates thinking from answering, the training hyperparameters, and the crucial choice of sequence length that affects test-time behavior.

- **Third, budget forcing — the core test-time intervention:** The mechanism for capping thinking (force-appending the end-of-thinking delimiter) and extending thinking (suppressing the delimiter, appending `"Wait"`). This is the paper's primary methodological contribution for achieving test-time scaling.

- **Fourth, the baseline methods for test-time compute control:** Token-conditional, step-conditional, class-conditional control, and rejection sampling, with an explanation of why each fails on controllability or scaling compared to budget forcing.

- **Fifth, the evaluation metrics for test-time scaling methods:** Control, Scaling, and Performance — the formal framework the paper introduces to compare methods quantitatively.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **methods paper** whose core idea is that test-time scaling behavior can be achieved through (a) supervised fine-tuning on a carefully selected small dataset of reasoning traces, and (b) a simple test-time token manipulation technique that controls reasoning duration with perfect precision. The contribution is not a new training algorithm but rather finding the minimal recipe that works, supported by systematic ablations.

---

#### Data Curation: From 59K to s1K

The data curation pipeline is a three-stage filtering process designed to select 1,000 training examples that jointly satisfy three criteria: **Quality** (no formatting errors or API failures), **Difficulty** (problems that challenge the target model), and **Diversity** (coverage across mathematical and scientific domains).

**Stage 0: Initial Distillation (59,029 → 54,116 → 51,581)**

The process begins by collecting 59,029 questions from 16 diverse sources. The largest sources are NuminaMATH (30,660 mathematical problems from online websites), MATH (11,999 competition math problems), OlympicArena (4,250 questions spanning Astronomy, Biology, Chemistry, Computer Science, Geography, Mathematics, and Physics from various Olympiads), OmniMath (4,238 competition-level mathematics problems), and AGIEval (2,385 questions from standardized tests like SAT and LSAT, covering English, Law, and Logic). The authors also create two original datasets: **s1-prob** (182 questions from Stanford University Statistics Department PhD Qualifying Exams, with handwritten solutions covering difficult proofs) and **s1-teasers** (23 challenging brain-teasers from quantitative trading interviews, selected at the highest difficulty level "Hard").

For each of these 59,029 questions, the authors generate a reasoning trace and solution using the **Google Gemini Flash Thinking API**, extracting the model's internal reasoning trace (the "thinking" part) and its final response. This yields triplets of (question, generated reasoning trace, generated solution). The authors then decontaminate all samples against evaluation questions (MATH500, GPQA Diamond, AIME24) using 8-gram overlap: any sample sharing more than 8 consecutive tokens with any evaluation question is excluded.

The first quality filter removes questions where API errors occurred (reducing the count from 59,029 to 54,116) and then removes low-quality examples containing formatting issues such as ASCII art diagrams, non-existent image references, or inconsistent question numbering (further reducing to 51,581). From this pool, the authors identify 384 samples from datasets they perceive as high-quality and not needing further filtering (these are correct AIME/GPQA solutions from Gemini, and correct MATH500 solutions with thinking traces longer than 5,600 tokens).

**Stage 1: Difficulty Filtering (51,581 → 24,496)**

The difficulty filter uses two imperfection signals: **model performance** and **reasoning trace length**. The authors evaluate two models on each question — Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct — with correctness assessed by Claude 3.5 Sonnet comparing each model's attempt against the Gemini-generated reference solution. The grading prompt (Figure 8 in the paper) asks Claude to compare the student's attempt with the correct answer, explaining its reasoning and outputting only "Yes" or "No" on a final line.

The key filtering criterion: **remove any question that either model can solve correctly.** If either Qwen2.5-7B-Instruct or Qwen2.5-32B-Instruct produces the correct answer, the question is deemed too easy and is excluded. Using two models reduces the likelihood of an easy sample slipping through due to a rare mistake on an otherwise easy question (both models would need to fail simultaneously for an easy question to survive). This brings the pool from 51,581 to 24,496 questions — all of which neither Qwen2.5-7B-Instruct nor Qwen2.5-32B-Instruct could solve correctly, implying they require reasoning beyond what the base models can do without test-time compute.

The reasoning trace length is also measured (using the Qwen2.5 tokenizer) as a secondary difficulty indicator, based on the assumption that more difficult problems require more thinking tokens. This length signal is used later in the diversity sampling stage to bias selection toward longer (harder) traces.

**Stage 2: Diversity-Aware Selection (24,496 → 1,000)**

The diversity stage uses a two-step sampling algorithm (Algorithm 1 in the paper) designed to select exactly 1,000 examples spanning 50 distinct mathematical and scientific domains while favoring more difficult (longer) reasoning traces within each domain.

First, the authors classify all 24,496 questions into domains using Claude 3.5 Sonnet, based on the **Mathematics Subject Classification (MSC)** system from the American Mathematical Society. The MSC taxonomy focuses on mathematics but also includes other sciences such as biology, physics, and economics. The resulting s1K dataset spans 51 domains, the largest being Geometry (109 questions, 560.2K total tokens), Number Theory (98 questions, 522.5K tokens), and Combinatorics (75 questions, 384.7K tokens). The full distribution is shown in Table 6 of the paper.

Then, the selection algorithm proceeds as follows until 1,000 examples are collected:

1. **Randomly select a domain** from the set of all available domains.
2. **Rank all questions** in that domain by their Gemini-generated reasoning trace length (in tokens).
3. **Apply power-law weighting** to sample one question from this domain, where the weight for the `$k$`-th longest trace is `$2^{-k}$`. This means the longest trace in a domain has weight `$2^{-1} = 0.5$`, the second-longest has weight `$2^{-2} = 0.25$`, the third-longest `$2^{-3} = 0.125$`, and so on. This creates a strong bias toward selecting questions with longer reasoning traces (higher perceived difficulty) while still allowing some stochasticity.
4. **Remove the selected question** from the domain's pool. If the domain becomes empty, remove it from the set of available domains.

This process ensures that the final 1,000 examples span 50 distinct domains (the 51 domains listed in Table 6, noting that 27 domains have ≤16 questions each) with a sampling bias toward harder problems within each domain. The combination of domain-balanced sampling (ensuring diversity across mathematical subfields) and length-weighted sampling (favoring difficult problems within each domain) operationalizes the joint Diversity + Difficulty criteria.

**What makes s1K special: the joint criteria.** The ablation in Table 2 validates that all three criteria are necessary. **1K-random** (only Quality, no Difficulty or Diversity filtering) achieves only 36.7% on AIME24 vs. 50.0% for s1K — a 13.3 percentage point drop. **1K-diverse** (uniform sampling across domains, no Difficulty weighting) achieves 26.7% — even worse. **1K-longest** (selecting the 1,000 samples with the longest reasoning traces, no Diversity constraint) achieves 33.3%, showing a boost on GPQA (59.6% vs. 57.6%) but still substantially worse overall. The key insight is that **long traces alone are not sufficient** — the longest traces may cluster in a few domains (e.g., difficult geometry problems), missing the breadth needed for generalization across diverse reasoning tasks. The joint criteria prevent this by ensuring domain coverage while favoring difficulty within each domain.

**A note on data correctness.** The Gemini-generated reasoning traces are not guaranteed to be correct. The authors' grader (Claude 3.5 Sonnet) deems 53.6% of the traces in s1K correct, and 63.0% in s1K-1.1 (the follow-up version using DeepSeek R1 traces — see §A). The authors deliberately include incorrect traces, focusing on "capturing the reasoning process rather than entirely correct solutions." This is a design choice: the model learns to produce long-form reasoning traces by imitating the *form* of reasoning, even when the content is sometimes incorrect. The actual correctness is then determined at test time through the reasoning process itself (potentially with self-correction via budget forcing).

---

#### Supervised Fine-Tuning: From Qwen2.5-32B-Instruct to s1-32B

The training procedure is intentionally minimal — designed to be "the simplest approach" for achieving test-time scaling.

**Base model.** The authors start with **Qwen2.5-32B-Instruct** (Qwen et al., 2024), a 32-billion parameter model that has already been pretrained and instruction-tuned. The choice of this specific model is motivated by its strong math performance: "on math tasks [it] generally matches or outperforms the larger Qwen2.5-72B-Instruct or other open models" (citing Dubey et al., 2024; Groeneveld et al., 2024; Muennighoff et al., 2024). Using an already instruction-tuned model (rather than a base pretrained model) means the model already understands the format of user-assistant interactions and has some reasoning capability; the fine-tuning only needs to teach the specific behavior of producing long reasoning traces within delimited sections.

**Token delimiter format.** The training data is formatted with special token delimiters that separate the reasoning ("thinking") stage from the answering stage:

```
<|im_start|>think
[reasoning trace]
<|im_start|>answer
[final answer]
```

The thinking stage is enclosed between `<|im_start|>think` and `<|im_start|>answer`, both preceded and followed by a newline. This explicit delimiter structure is crucial for budget forcing at test time: the system can monitor for the `<|im_start|>answer` token to detect when the model wants to stop thinking, and can either suppress it (to extend thinking) or force it (to cap thinking).

The loss is computed only on the reasoning traces and solutions — **not on the questions themselves.** This means the model receives gradient updates only for generating the thinking process and final answer, not for predicting the question text. This focuses the training signal on the behavior of interest: producing long reasoning chains followed by answers.

**Training hyperparameters.** The training uses standard settings, chosen for simplicity rather than through extensive hyperparameter optimization:

- **Epochs:** 5
- **Batch size:** 16 (effective batch size; trained on 16 H100 GPUs with PyTorch FSDP)
- **Total gradient steps:** 315 (since 1,000 samples ÷ 16 batch size = 62.5 steps per epoch × 5 epochs ≈ 315 steps)
- **Precision:** bfloat16
- **Learning rate:** `$1 \times 10^{-5}$`, warmed up linearly for 5% of training (16 steps), then decayed to 0 over the remaining 95% of training (299 steps) following a cosine schedule
- **Optimizer:** AdamW (Loshchilov & Hutter, 2019) with `$\beta_1 = 0.9$`, `$\beta_2 = 0.95$`, and weight decay of `$1 \times 10^{-4}$`
- **Training duration:** 26 minutes on 16 NVIDIA H100 GPUs (totaling ~7 H100 GPU hours)

Compare this to training on the full 59K dataset, which requires 394 H100 GPU hours — a ~56× increase in compute for only modest performance gains (53.3% vs. 50.0% on AIME24, as shown in Table 2, with the 95% confidence interval for the difference being [-13.3%, +20.0%] — meaning we cannot confidently say 59K-full is better than s1K).

**The crucial sequence length ablation.** The main training hyperparameter that the authors ablate is the **training sequence length** (Table 8). This choice has a non-obvious and important effect on test-time behavior:

- **Short sequence length (4,096 tokens):** 74% of training samples are cut off (truncated). At test time, the model generates longer reasoning traces (20,721 average thinking tokens on AIME24) but achieves lower accuracy (30.0% on AIME24).

- **Long sequence length (32,768 tokens):** 0% of training samples are cut off. At test time, the model generates shorter reasoning traces (6,984 average thinking tokens on AIME24) and achieves higher accuracy (50.0% on AIME24).

The mechanism behind this counterintuitive result: **when training with a shorter sequence length, the answer section of training samples is more commonly cut off.** This means the model receives fewer gradient updates where it learns to actually transition from thinking to answering — it sees many examples of reasoning that end without ever reaching the answer. Consequently, at test time, the model has a lower log-probability of transitioning to the answer section, leading it to continue reasoning for longer before finally producing an answer (longer reasoning traces) but with lower overall quality (worse accuracy). Conversely, training with a long sequence length ensures that nearly every training example includes the full thinking → answer transition, giving the model strong gradients to learn when and how to conclude its reasoning and produce a final answer. This results in shorter, more efficient reasoning at test time with better accuracy.

The authors choose the longest training sequence length (32,768 tokens) because "it leads to better performance and makes inference more efficient by leading to shorter reasoning traces."

**The superficial alignment hypothesis for reasoning.** The authors hypothesize that supervised fine-tuning on just 1,000 examples works because the model already possesses reasoning capabilities from pretraining on trillions of tokens:

> "We hypothesize that the model is already exposed to large amounts of reasoning data during pretraining which spans trillions of tokens. Thus, the ability to perform reasoning is already present in our model. Our sample-efficient finetuning stage just activates it and we scale it further at test time with budget forcing. This is similar to the 'Superficial Alignment Hypothesis' presented in LIMA (Zhou et al., 2023)."

In this view, fine-tuning on s1K does not teach the model *how to reason* — it already knows how to reason from pretraining. Instead, fine-tuning teaches the model a specific format and behavior: (a) to produce long-form reasoning traces before answering, (b) to separate thinking from answering with explicit delimiters, and (c) to generate structured, step-by-step reasoning appropriate for mathematical and scientific problems. The extreme sample efficiency (1,000 vs. ≫800K for r1-distill) is evidence for this view: if the model were learning reasoning from scratch, far more examples would be needed.

---

#### Budget Forcing: The Core Test-Time Intervention

Budget forcing is the paper's primary methodological contribution for achieving **test-time scaling** — the ability to control and vary the amount of test-time computation the model spends on reasoning, with corresponding changes in accuracy. It is a decoding-time intervention that manipulates the model's generation at the token level, requiring no additional training, no auxiliary models, and no changes to the model weights.

The mechanism exploits the explicit token delimiter structure established during fine-tuning. Recall that the model is trained to produce:

```
<|im_start|>think
[reasoning trace tokens]
<|im_start|>answer
[final answer tokens]
```

The `<|im_start|>answer` delimiter is the key control point. It signals the transition from thinking to answering. Budget forcing works by manipulating whether and when this delimiter appears.

**Operation I: Capping Thinking (Enforcing a Maximum Budget)**

To force the model to think for *at most* a specified number of tokens, the system simply monitors the generation and, when the token count reaches the limit, forcibly appends the end-of-thinking token delimiter and optionally `"Final Answer:"` to the model's current generation. This causes the model to immediately transition from thinking mode to answering mode, producing its current best answer.

The mechanism is: during autoregressive generation, the model normally samples the next token from its output distribution at each step. When the token budget is exhausted, the system bypasses the model's sampling and directly concatenates the delimiter string to the generation. The model then continues generating from this new context, but now "sees" that the thinking stage has ended and it should produce an answer.

This is illustrated in the left panels of Figure 4: for s1-32B, capping thinking at 2,048 or 4,096 tokens produces a clear accuracy vs. token budget relationship — more tokens allocated to thinking yields higher accuracy. The "Forcing 2048/4096 max thinking tokens" label in Figure 4(a) refers to this operation.

**Operation II: Extending Thinking (Enforcing a Minimum Budget)**

To force the model to think *longer* than it naturally would, the system monitors the generation for the end-of-thinking token delimiter (`<|im_start|>answer`). When the model attempts to generate this token (indicating it "wants" to stop thinking and start answering), the system **suppresses** that token — it does not add it to the generated sequence — and instead appends the string `"Wait"` to the model's current reasoning trace.

The mechanism is more subtle than simple capping. During autoregressive generation, the model's output distribution at each step produces a probability for each token in the vocabulary. When the model assigns high probability to the `<|im_start|>answer` token, it is signaling completion. The system detects this, prevents that token from being emitted, and instead forces the string `"Wait"` into the context. The model then continues generating from this modified context, but now "sees" that it has been told to wait, which encourages it to continue reasoning — often by double-checking its work, reconsidering assumptions, or exploring alternative solution paths.

Figure 3 shows a concrete example of this self-correction in action. The model initially counts the number of 'r's in "raspberry":
- First pass: counts 2 'r's (incorrect — there are actually 3)
- The model attempts to end thinking with its final answer being 2
- The system suppresses the end-of-thinking delimiter and appends `"Wait"`
- The model re-reads the question, identifies its mistake, and correctly counts 3 'r's
- The model then produces the correct answer

The appending of `"Wait"` can be done multiple times — each time the model tries to stop thinking, the system intervenes. Figure 4(a) shows this: for the three rightmost data points, the system prevents the model from stopping its thinking 2, 4, and 6 times respectively, each time appending `"Wait"` to the current reasoning trace. The model can be forced to think for up to ~6 interventions before the behavior degrades into repetitive loops.

**Why "Wait" specifically?** Table 4 ablates the choice of appended string when suppressing the end-of-thinking delimiter twice (2x). The options compared are:
- **No string appended (2x without string):** 50.0% on AIME24, 90.2% on MATH500, 55.1% on GPQA
- **"Alternatively":** 50.0% on AIME24, 92.2% on MATH500, 59.6% on GPQA
- **"Hmm":** 50.0% on AIME24, 93.0% on MATH500, 59.6% on GPQA
- **"Wait":** 53.3% on AIME24, 93.0% on MATH500, 59.6% on GPQA

"Wait" provides the strongest AIME24 performance (53.3% vs. 50.0% for "Hmm" and "Alternatively" and no string). On MATH500, "Hmm" and "Wait" tie at 93.0%. On GPQA, all three strings achieve 59.6%, beating no string (55.1%). The mechanism is likely that "Wait" is a natural continuation in English discourse when someone realizes they need to reconsider — it signals "stop, I need to think more carefully" — and the model's pretraining on human text has encoded this convention.

**The limits of budget forcing for extending thinking.** The paper is explicit about two key limitations:

1. **Flattening at ~6 interventions:** Suppressing the end-of-thinking delimiter too many times leads the model into repetitive loops rather than continued productive reasoning. The scaling curve in Figure 4(a) "does eventually flatten out at six times." This is likely because the model's context becomes saturated with previous reasoning attempts and "Wait" signals, and the model loses the ability to generate novel, useful reasoning steps.

2. **Context window limitations:** At some point, the accumulated reasoning trace exceeds the model's maximum context length. The authors note this explicitly for the step-conditional experiments (where 12 out of 30 AIME24 questions exceeded the context window when using up to 512 steps), and the same constraint applies to budget forcing with repeated "Wait" interventions.

**Perfect controllability.** Unlike conditional control methods (token, step, class), budget forcing achieves **100% Control** as defined by the paper's metrics (Equation 1). The Control metric measures the fraction of evaluation runs where the actual thinking token count falls within a pre-specified range. Because budget forcing directly manipulates tokens — either forcibly stopping generation at a precise limit or forcibly extending generation by a precise number of interventions — it achieves perfect adherence to the desired budget. This is a significant advantage over methods that rely on the model "choosing" to comply with length instructions (which it often fails to do).

**Test-time scaling with budget forcing.** Figure 1 shows the core result: as more test-time compute is allocated (x-axis: average thinking tokens), s1-32B's accuracy improves. On MATH500, accuracy rises from ~65% at 512 tokens to ~85% at 2,048 tokens. On AIME24, accuracy rises from ~20% at 1,024 tokens to 57% at 7,320 tokens (the rightmost dot, corresponding to 6 "Wait" interventions). On GPQA Diamond, accuracy rises from ~40% at 1,024 tokens to ~60% at 4,096 tokens.

Figure 4(b) provides the crucial comparison: **sequential scaling (budget forcing with s1-32B) vs. parallel scaling (majority voting with the base Qwen2.5-32B-Instruct).** Majority voting with up to 64 parallel samples achieves only ~60% accuracy on the y-axis scale shown, while sequential scaling with s1-32B achieves higher performance. The key insight is that sequential scaling operates in a "different scaling paradigm" — it has a steeper slope and extrapolates better because "later computations can build on intermediate results, allowing for deeper reasoning and iterative refinement" (Section 3.1). Parallel sampling, by contrast, treats each attempt as independent and only aggregates at the end, losing the benefit of iterative refinement.

---

#### Baseline Methods for Test-Time Compute Control

The paper compares budget forcing against four alternative approaches for controlling test-time compute, each of which has significant limitations that budget forcing overcomes.

**Token-Conditional Control (Specifying Token Budgets in the Prompt)**

The idea is to tell the model in the user prompt exactly how many thinking tokens it should generate: "Think for up to 2048 tokens." The training data is bucketed by reasoning trace length into powers of two (rounded upward), and the corresponding token instruction is added to each training example's user prompt. For example, a training sample with a reasoning trace between 1,024 and 2,048 tokens gets the instruction "Think for up to 2048 tokens."

**Why it fails:** Table 12 shows that the model does not reliably follow token instructions. When instructed to think for 1,024 tokens, it generates 7,939 thinking tokens (nearly 8× the limit). When instructed for 2,048 tokens, it generates 7,158. When instructed for 16,384 tokens, it generates 7,500. The model's generation length is largely insensitive to the instruction — it generates roughly the same number of tokens regardless. This is consistent with prior findings that even OpenAI's o1-mini cannot follow token-length instructions (Zhang & Chen, 2024).

Combining token-conditional control with budget forcing (capping thinking when the limit is reached) solves the control problem: the model generates until the limit, then is forced to stop. This achieves 100% Control, but the Performance is still lower than budget forcing alone (40.0% vs. 56.7% on AIME24 in Table 3). This may be because the token instruction in the prompt interferes with the model's natural reasoning process, or because training with token instructions is a weaker signal than simply letting the model generate freely and then capping it.

**Step-Conditional Control (Specifying Step Budgets in the Prompt)**

The idea is to make counting more coarse-grained by partitioning reasoning traces into "steps" (separated by double newlines) and asking the model to count down steps: "Think for up to 64 steps" with the model generating "64 steps left... 63 steps left..." at each step break. The format (Figure 10, right) was chosen based on early experiments showing the model was more likely to adhere to the limit when counting down rather than up — likely because when counting down, the final step is always 1, providing a strong signal to conclude.

**Why it fails:** Table 13 reveals two problems:

1. **The model compensates.** When forced to use up to 16 steps, the model generates an average of 96 tokens per step; when given 256 steps, it generates only 56 tokens per step. The model "hacks its way around the compute constraint" by making each step shorter when more steps are required, keeping total thinking tokens roughly constant (7,252 at 16 steps, 7,551 at 256 steps — nearly identical).

2. **Step delimiters are costly.** Each step delimiter ("64 steps left") requires ~6 tokens. For 64 steps, this adds ~380 tokens of overhead that do not contribute to reasoning. Even ignoring these overhead tokens in the counts, the step-conditional model requires 7,551 thinking tokens to achieve only 33.3% on AIME24 — far less efficient than budget forcing.

Combining step-conditional control with budget forcing (capping when 0 steps are reached) solves the control issue but, as with token-conditional control, Performance remains low (36.7% on AIME24 in Table 3).

**Class-Conditional Control (Vague "Think Longer" Prompts)**

The idea is inspired by OpenAI's "reasoning_effort" API parameter (low/medium/high). The authors test two generic prompts appended to the question: "Answer after a short amount of thinking. Do not spend excessive time double-checking your work." and "Answer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to double-check your work until you are absolutely sure that you have the correct answer."

**Results (Table 14):** The long-thinking prompt does increase thinking tokens (9,651 vs. 8,033 for short-thinking on AIME24), but performance is inconsistent: on AIME24, long-thinking achieves 36.7% vs. 30.0% for short-thinking (an improvement), but on GPQA, long-thinking achieves 51.0% vs. 56.6% for short-thinking (a degradation). Importantly, both class-conditional prompts underperform the baseline model without any generic prompt (50.0% on AIME24, 93.0% on MATH500, 57.6% on GPQA). The Control metric is only 50%: in one of the two desired directions (short should produce fewer tokens than default, long should produce more), the model does not comply.

**Rejection Sampling (Sample Until Length Fits)**

The idea is to sample generations from the model (at temperature 1) until a generation naturally has a reasoning trace shorter than a specified length limit. This is an oracle method that captures the posterior distribution over responses conditioned on length.

**Why it fails (Figure 6, Table 3):** Rejection sampling shows **inverse scaling** — accuracy *decreases* as the allowed thinking token budget increases. At ≤3,500 tokens (requiring an average of 655 attempts per sample), accuracy is ~42%. At ≤16,000 tokens (requiring only 1 attempt on average), accuracy drops to ~32%. The Scaling metric is -35 in Table 3 — strongly negative.

The authors hypothesize a correlation: "shorter generations tend to be the ones where the model was on the right track from the start, whereas longer ones tend to be ones where the model made mistakes and thus backtracks or questions itself." In other words, *correct reasoning is often more direct*, and forcing the model to limit its thinking selects for the more direct (and more often correct) solutions. Longer reasoning traces, when naturally generated, are often longer *because* the model is confused and going in circles. This is a critical negative result: you cannot simply "let the model think longer" by sampling until it produces a long trace — the long traces are the wrong ones.

An example in §E.2 illustrates this: the same AIME question produces a correct answer when the model is constrained to ≤4,000 thinking tokens (the model takes a direct, correct approach) but an incorrect answer when allowed ≤8,000 thinking tokens (the model backtracks extensively, gets confused, and converges on a wrong answer).

**Why budget forcing wins.** Table 3 summarizes the comparison across the three evaluation metrics:

| Method | Control | Scaling | Performance | |𝒜| |
|--------|---------|---------|-------------|---|
| Budget Forcing | 100% | 15 | 56.7 | 5 |
| Token-Conditional | 40% | -24 | 40.0 | 5 |
| Token-Conditional + BF | 100% | 13 | 40.0 | 5 |
| Step-Conditional | 60% | 3 | 36.7 | 5 |
| Step-Conditional + BF | 100% | 6 | 36.7 | 5 |
| Class-Conditional | 50% | 25 | 36.7 | 2 |
| Rejection Sampling | 100% | -35 | 40.0 | 5 |

Budget forcing dominates on Control (100%, tied only with methods that also use forcing) and Performance (56.7%, substantially higher than all alternatives). Its Scaling of 15 is positive and substantial, though class-conditional control has a higher Scaling of 25 — but this is based on only 2 evaluation runs (|𝒜|=2) and with a Control of only 50%. The key advantage is that budget forcing achieves all three desiderata simultaneously: perfect control, positive scaling, and strong absolute performance.

---

#### Evaluation Metrics for Test-Time Scaling Methods (Section 3.2)

The paper introduces three quantitative metrics to formally compare test-time scaling methods. These metrics are applied to a set of evaluation runs `$\mathcal{A}$` where test-time compute is varied and accuracy is measured on a fixed benchmark.

**Notation.** For a given method and benchmark, the evaluation produces a piecewise linear function `$f$` mapping test-time compute (measured in thinking tokens, on the x-axis) to accuracy (as a percentage, on the y-axis). Each evaluation run `$a \in \mathcal{A}$` has a specific amount of test-time compute, producing an accuracy `$f(a)$`. The set `$\mathcal{A}$` contains all the test-time compute levels evaluated.

**Metric 1: Control (Equation 1)**

$$\text{Control} = \frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} \mathbb{I}(a_{\min} \leq a \leq a_{\max})$$

where `$a_{\min}$` and `$a_{\max}$` are pre-specified minimum and maximum amounts of test-time compute (in thinking tokens), `$a$` is the actual amount of test-time compute for a given evaluation run, `$\mathbb{I}(\cdot)$` is the indicator function (1 if the condition is true, 0 otherwise), and `$|\mathcal{A}|$` is the total number of evaluation runs.

**What it computes:** the fraction of evaluation runs where the actual test-time compute falls within the desired range. For capping methods, this measures whether the model stays under the specified limit. For extending methods, this measures whether the model reaches the specified minimum. A value of 100% means the method perfectly controls test-time compute — every run respects the limits.

**Why this form:** the indicator function provides a binary pass/fail for each run, and the average gives an interpretable percentage. Alternative formulations like "average distance from target" would conflate overshooting and undershooting and would be harder to interpret. The binary indicator directly measures whether the method achieves its stated goal.

**Metric 2: Scaling (Equation 2)**

$$\text{Scaling} = \frac{1}{\binom{|\mathcal{A}|}{2}} \sum_{a, b \in \mathcal{A}, b > a} \frac{f(b) - f(a)}{b - a}$$

where `$f(a)$` is the accuracy at test-time compute level `$a$`, `$f(b)$` is the accuracy at a higher test-time compute level `$b$`, and `$\binom{|\mathcal{A}|}{2}$` is the number of pairs of distinct evaluation runs.

**What it computes:** the average slope of the piecewise linear function `$f$` across all pairs of evaluation runs where one has more test-time compute than the other. For each pair `$(a, b)$` with `$b > a$`, it computes the slope `$(f(b) - f(a)) / (b - a)$` — the change in accuracy per unit change in thinking tokens. It then averages these slopes over all pairs. A positive value means that, on average, allocating more test-time compute leads to higher accuracy. A negative value means that more test-time compute *decreases* accuracy (inverse scaling). Larger positive values indicate steeper scaling — more accuracy gain per additional token of thinking.

**Why this form:** the average slope across all pairs provides a summary statistic that captures the overall trend, rather than being sensitive to any single budget choice. A method with consistently positive marginal returns will have a high positive Scaling; a method where some budget increases help but others hurt will have a lower average; a method with inverse scaling will be negative. The pairwise formulation uses all available information from the evaluation runs rather than just comparing the minimum and maximum budgets. The denominator `$\binom{|\mathcal{A}|}{2}$` ensures the metric is normalized to be independent of the number of evaluation runs.

**Metric 3: Performance (Equation 3)**

$$\text{Performance} = \max_{a \in \mathcal{A}} f(a)$$

where `$f(a)$` is the accuracy at test-time compute level `$a$`.

**What it computes:** the maximum accuracy achieved by the method on the benchmark across all tested test-time compute levels. This is the simplest metric: "how good can this method get, if we're allowed to choose the best budget?"

**Why this form:** maximum performance captures the ceiling of the method — its potential if the budget is chosen optimally. A method that scales well but from a low base might have high Scaling but low Performance; a method that doesn't scale but has high baseline performance might have low Scaling but high Performance. Together, Control, Scaling, and Performance provide a three-dimensional view of method quality: can you control it? does more compute help? how good can it get?

**Interpreting the metrics.** A method like rejection sampling achieves 100% Control (you can always sample until the length fits) and moderate Performance (40.0%) but *negative* Scaling (-35) — additional compute makes it worse. A method like class-conditional control achieves positive Scaling (25) but poor Control (50%) and modest Performance (36.7%). Budget forcing achieves the trifecta: 100% Control, positive Scaling (15), and the best Performance (56.7%). This three-metric evaluation framework is one of the paper's contributions, providing a standardized way to compare future test-time scaling methods.

## 4. Key Insights and Innovations

### Innovation 1: Test-Time Scaling Through Pure Supervised Fine-Tuning, Not Reinforcement Learning

The dominant narrative around test-time scaling — the phenomenon where a language model's performance improves predictably with additional inference computation — has been that it requires **reinforcement learning (RL)**. OpenAI described o1's training as "large-scale reinforcement learning" (OpenAI, 2024). DeepSeek R1 (DeepSeek-AI et al., 2025) and Kimi k1.5 (Team et al., 2025) both employed massive RL pipelines with millions of training samples and multiple training stages. The field had converged on the assumption that test-time scaling behavior emerges from the model learning to *value* its own intermediate reasoning steps — that the model needs reward signals to develop the capacity to allocate more thinking to harder problems and to systematically improve with additional compute.

This paper **breaks that assumption**. It demonstrates that supervised fine-tuning (SFT) on just 1,000 carefully selected examples — with no reinforcement learning, no process reward models, no value functions — produces a model (s1-32B) that exhibits clean test-time scaling behavior across three benchmarks (Figure 1). The scaling curves are not as steep as o1's reported curves (and the absolute performance is lower than full o1), but they are unambiguous: accuracy on AIME24 rises from ~20% at 1,024 thinking tokens to 57% at 7,320 tokens; on MATH500 from ~65% at 512 tokens to ~85% at 2,048 tokens; on GPQA Diamond from ~40% at 1,024 tokens to ~60% at 4,096 tokens.

What makes this a conceptual advance rather than merely an incremental engineering finding is that it **reframes the nature of reasoning capability in language models**. The paper's explicit reference to the "Superficial Alignment Hypothesis" from LIMA (Zhou et al., 2023) extends that hypothesis from instruction-following to the much more demanding domain of multi-step mathematical and scientific reasoning. The implication is that the capacity for reasoning — including the ability to allocate more computation to harder problems and to improve with extended thinking — is **already latent in pretrained models** from their exposure to trillions of tokens of text during pretraining. The fine-tuning stage does not *teach* reasoning; it *elicits* it by showing the model the desired format: produce long, structured reasoning traces before answering, separated by explicit delimiters. The test-time intervention (budget forcing) then manipulates this format to control how much of the latent reasoning capability is deployed.

This is a fundamental shift from the RL-centric view. If correct, it means the bottleneck for building reasoning models is not capability *acquisition* (which requires massive RL) but capability *elicitation* (which can be achieved through sample-efficient SFT and inference-time interventions). The extreme sample efficiency — 1,000 examples versus the ≫800K used for r1-distill — is not just a practical convenience but a **scientific signal** supporting this reframing. The ablation in Table 2 showing that training on the full 59K dataset produces only marginal gains over the 1K subset (53.3% vs. 50.0% on AIME24, with a 95% confidence interval that cannot exclude zero difference) further supports the view that more data does not substantially augment the underlying capability — the pretrained model already has what it needs.

A critical nuance: this finding does **not** claim that SFT is *superior* to RL for building reasoning models. The r1-distill model trained on 800K SFT examples from DeepSeek R1 achieves 72.6% on AIME24 (Table 1), substantially higher than s1-32B's 56.7%. The claim is rather that **RL is not necessary for the core phenomenon of test-time scaling** — that the scaling behavior itself can emerge from simple SFT with the right data formatting, even if RL can push absolute performance higher.

### Innovation 2: Budget Forcing as a Token-Level Mechanism for Perfect Compute Controllability

Prior work on controlling test-time compute in language models relied on **prompt-based instructions** (telling the model to "think longer" or "think for up to N steps") or **post-hoc selection** (rejection sampling, best-of-N, majority voting). None of these achieved both reliable control and positive scaling simultaneously. Token-conditional control fails because models cannot count tokens. Step-conditional control fails because models compensate by varying tokens-per-step, keeping total computation roughly constant (Table 13). Class-conditional control provides only coarse separation with unreliable performance improvement (Table 14). Rejection sampling produces *inverse* scaling — longer natural generations are less accurate (Figure 6, Scaling = -35 in Table 3).

Budget forcing introduces a fundamentally different control mechanism: **direct token manipulation at the delimiter level**. Instead of asking the model to comply with a length instruction (which it cannot do reliably), budget forcing intervenes in the generation process itself by suppressing or injecting specific delimiter tokens. The key insight is that by establishing an explicit `<|im_start|>answer` delimiter during fine-tuning — a token that the model learns signals the transition from reasoning to answering — the system gains a precise **control handle** at test time. Suppressing this token prevents the transition; injecting it forces the transition. The model does not need to count tokens or follow instructions; it simply continues its autoregressive generation from whatever context the system provides.

What makes this distinctive at the idea level is that it **treats the delimiter not as passive formatting but as an active control interface**. In standard instruction tuning, delimiters like `<|im_start|>assistant` are structural markers — the model learns to produce them at appropriate points, but they are not manipulated at test time. Budget forcing repurposes these delimiters as **gates** that the system can open or close to precisely regulate reasoning duration. This is conceptually similar to how an operating system controls process execution through interrupts and signals, rather than asking processes to voluntarily yield.

The formal evaluation framework introduced in Section 3.2 — measuring Control, Scaling, and Performance as three orthogonal dimensions of test-time scaling method quality — provides the language to articulate why budget forcing is distinctive. It achieves **100% Control** (every evaluation run respects the specified budget), **positive Scaling** (15 on AIME24, the highest among methods with adequate control), and the **best Performance** (56.7%, substantially above alternatives). No prior method achieves this trifecta. The framework itself is a contribution — it moves the discussion from qualitative claims ("this method scales well") to quantitative comparison, enabling future researchers to evaluate new test-time scaling methods against a clear, multidimensional standard.

The discovery that "Wait" is the most effective string to append when suppressing the delimiter (Table 4) is a minor but revealing detail. The alternatives — "Hmm," "Alternatively," or no string — all produce weaker or equal results. This suggests that the effectiveness of budget forcing depends on tapping into the model's pretrained discourse patterns — "Wait" is a natural marker of realization and reconsideration in English dialogue, and the model's pretraining has encoded that appending "Wait" should trigger re-examination of previous statements.

### Innovation 3: The Inverse Scaling of Rejection Sampling as a Diagnostic Finding

Rejection sampling — generating multiple solutions and selecting those whose length fits a desired budget — seems intuitively like a natural approach to controlling test-time compute. If you want longer reasoning, sample until you get a naturally long reasoning trace. The approach requires no training, no architecture changes, and works with any model.

The paper's finding that rejection sampling produces **inverse scaling** (Figure 6, Table 3: Scaling = -35) is a **diagnostic negative result** with significant implications for how the field thinks about test-time compute. Accuracy on AIME24 *decreases* from ~42% to ~32% as the allowed thinking token budget increases from ≤3,500 to ≤16,000 tokens. The mechanism: shorter natural generations tend to be the ones where the model was on the right track from the start and reached the answer efficiently; longer natural generations tend to be longer *because* the model is confused, backtracking, and questioning itself. The correlation between generation length and correctness is **negative** for naturally generated traces.

This finding matters because it **inverts the intuitive relationship between thinking time and accuracy**. The default assumption — which motivated many prior approaches to test-time scaling — is that "more thinking equals better answers." Rejection sampling shows that this is true only when thinking time is *controlled externally* (through budget forcing) rather than *self-selected* by the model. When the model chooses its own thinking duration, longer is often worse. This is a critical caution for anyone building test-time scaling systems: you cannot simply let the model think as long as it wants and expect improvement. You need an external mechanism (like budget forcing) that *forces* additional thinking in a way that produces productive reconsideration rather than amplifying confusion.

The qualitative example in §E.2 of the paper illustrates this precisely: the same AIME problem produces a correct answer when the model's thinking is capped at 4,000 tokens (the model takes a direct, correct approach) but an incorrect answer when allowed 8,000 tokens (the model backtracks extensively, explores dead ends, and converges on a wrong answer). This is not an anomaly — it is the systematic pattern that drives the negative scaling slope.

Conceptually, this finding draws a sharp distinction between **allocated compute** (externally imposed by budget forcing) and **emergent compute** (the model's natural tendency to think longer on problems where it struggles). These two quantities are negatively correlated in the natural generation distribution, and only by breaking this correlation — through external control — can test-time scaling become productive. This insight is complementary to prior work on "let's verify step by step" (Lightman et al., 2023) and process reward models, which focused on *evaluating* reasoning quality, not on *controlling* reasoning duration.

### Innovation 4: Joint Difficulty-Diversity-Quality Criteria as the Key to Sample-Efficient Reasoning Training

The paper's data ablation (Table 2) is not merely a validation of s1K — it is a **conceptual contribution about what makes reasoning training data effective**. The three ablations isolate each criterion:

- **1K-random** (only Quality, no Difficulty or Diversity filtering): 36.7% on AIME24 — a 13.3 percentage point drop from s1K's 50.0%. This shows that random selection from a high-quality pool is insufficient; some examples are too easy or too repetitious to teach the model the desired behavior.

- **1K-diverse** (uniform sampling across domains, no Difficulty weighting): 26.7% on AIME24 — even worse than random. This shows that diversity alone is harmful if it includes easy examples; the model learns to produce reasoning traces for simple problems that don't require the kind of extended, self-correcting reasoning needed for competition math.

- **1K-longest** (the 1,000 samples with longest reasoning traces, no Diversity constraint): 33.3% on AIME24. This boosts GPQA performance (59.6% vs. 57.6% for s1K), likely because long traces on science problems teach more thorough reasoning, but overall performance still substantially lags s1K. The mechanism: the longest traces tend to cluster in a few domains (e.g., certain types of geometry problems), leaving the model undertrained on other reasoning domains.

The key conceptual finding is that **no single criterion dominates, and criteria interact multiplicatively**. Long traces are good, but only when they span diverse domains. Diversity is good, but only when the examples are sufficiently difficult. Quality is necessary but not sufficient. The joint application of all three — selecting examples that a weaker model *cannot* solve (Difficulty), spanning 50 distinct mathematical subfields (Diversity), and filtering out formatting errors (Quality) — produces a dataset that is dramatically more effective per example than any subset chosen by a single criterion.

This finding connects to and extends the instruction tuning literature. Zhou et al. (2023) showed that 1,000 high-quality examples sufficed for alignment. But their criteria for "high-quality" were holistic and somewhat subjective. This paper **operationalizes quality for reasoning** as the joint product of difficulty (measured by model failure), diversity (measured by domain classification and balanced sampling), and basic formatting standards. This provides a recipe that others can follow, rather than leaving data curation as an art. The fact that training on the full 59K dataset — a superset containing all ablation subsets — produces only marginal gains over the 1K selection (53.3% vs. 50.0% on AIME24, with overlapping confidence intervals) demonstrates that **the selection algorithm, not the total data volume, is the primary driver of sample efficiency**.

The difficulty filter deserves particular attention as a methodological innovation. By requiring that *both* Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct fail to solve a problem, the filter selects for problems that are genuinely beyond the model's current one-shot capability — precisely the problems where extended reasoning and self-correction are needed. This creates a training distribution that teaches the model to reason about problems it cannot solve immediately, which is exactly the behavior that test-time scaling amplifies. This is a concrete operationalization of the "zone of proximal development" concept — select training examples at the frontier of the model's current abilities, where additional reasoning effort can make the difference between failure and success.

The power-law sampling within each domain (Algorithm 1, with weight `$2^{-k}$` for the `$k$`-th longest trace) is also a subtle but important design choice. It biases selection strongly toward harder problems while maintaining some stochasticity, preventing the model from overfitting to the single hardest problem in each domain. The base-2 exponential weighting means the top 3-4 longest traces in each domain have a reasonable chance of selection, while traces beyond the top ~10 are essentially never selected. This balances the difficulty signal with some variety in problem types within each domain.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All experiments use three benchmarks: (1) **AIME24** — 30 problems from the 2024 American Invitational Mathematics Examination, requiring integer answers from 000–999 and testing arithmetic, algebra, counting, geometry, number theory, and probability; (2) **MATH500** — the same 500 competition math problems selected by OpenAI in Lightman et al. (2023) from the original MATH benchmark (Hendrycks et al., 2021), of varying difficulty; (3) **GPQA Diamond** — 198 PhD-level science questions from Biology, Chemistry, and Physics (Rein et al., 2023), where domain experts with PhDs achieved only 69.7% accuracy. These three benchmarks were chosen to span three distinct reasoning intensity levels: competition math (AIME24), general math (MATH500), and graduate-level science (GPQA Diamond).

- **Base model(s).** All experiments start from **Qwen2.5-32B-Instruct** (Qwen et al., 2024), a 32-billion parameter pretrained and instruction-tuned model. The authors state this model was chosen because "on math tasks [it] generally matches or outperforms the larger Qwen2.5-72B-Instruct or other open models" (citing Dubey et al., 2024; Groeneveld et al., 2024; Muennighoff et al., 2024). For the parallel scaling comparison in Figure 4(b), the un-fine-tuned Qwen2.5-32B-Instruct serves as the base model for majority voting. For the REBASE comparison in Figure 7, the authors use the REBASE process reward model initialized from LLaMA-34B and further fine-tuned on a synthetic process reward modeling dataset (Wu et al., 2024b). Comparison models include: o1-preview, o1-mini, o1 (OpenAI, 2024); Gemini 2.0 Flash Thinking Experimental (Google, 2024); DeepSeek r1 and r1-distill (DeepSeek-AI et al., 2025); QwQ-32B-preview (Team, 2024); Sky-T1-32B-Preview (Team, 2025); Bespoke-32B (Labs, 2025); and the follow-up models o3-mini-low/medium/high (OpenAI, 2025) and LIMO (Ye et al., 2025a).

- **Metrics.** The primary metric throughout is **accuracy** (equivalent to pass@1), measured as the fraction of questions where the model's final answer matches the ground truth. For AIME24, answers are integers 000–999; grading uses exact matching. For MATH500 and GPQA Diamond, the lm-evaluation-harness framework (Gao et al., 2021; Biderman et al., 2024) handles answer extraction and comparison. Unless otherwise specified, evaluations run with temperature 0 (greedy decoding). For test-time scaling comparisons specifically, the paper introduces three additional metrics: **Control** (Equation 1 — fraction of evaluation runs where actual thinking tokens fall within a pre-specified range), **Scaling** (Equation 2 — average slope of accuracy vs. thinking tokens across all pairs of evaluation runs), and **Performance** (Equation 3 — maximum accuracy achieved across all tested compute levels). These are reported as dimensionless quantities for Control (percentage) and Scaling (units of accuracy percentage per thinking token, though the paper reports them as raw numbers). For the sequential vs. parallel experiments, the x-axis is **average thinking tokens** — the mean number of tokens generated in the reasoning trace across all evaluated questions at a given budget setting. For majority voting, "thinking tokens" includes all tokens across all parallel samples, making the comparison to sequential scaling fair in terms of total FLOPs.

- **Baselines.** The paper compares s1-32B against several categories of baselines: (1) **API-only models** — o1-preview, o1-mini, o1, Gemini 2.0 Flash Thinking Experimental; (2) **Open-weight models without open data** — the base Qwen2.5-32B-Instruct, QwQ-32B-preview, DeepSeek r1, r1-distill (trained on 800K examples); (3) **Open-weight models with open data** — Sky-T1-32B-Preview (17K examples), Bespoke-32B (17K examples); (4) **Ablated versions of s1-32B** — the model without budget forcing (s1 w/o BF), and models trained on alternative data selections (1K-random, 1K-diverse, 1K-longest, 59K-full). For test-time scaling method comparisons (Table 3), the baselines are: Token-Conditional Control (TCC), Step-Conditional Control (SCC), Class-Conditional Control (CCC), and Rejection Sampling (RS), each evaluated with and without budget forcing augmentation where applicable.

- **Generation budget / compute accounting.** The universal unit of test-time compute is **thinking tokens** — the number of tokens generated in the reasoning trace before the answer delimiter. This is directly measured from model generations using the Qwen2.5 tokenizer. For sequential scaling (budget forcing), thinking tokens include all tokens generated during the extended reasoning, including any forced "Wait" tokens appended by the system. For parallel scaling (majority voting), the total thinking tokens are the sum across all parallel samples — for 64 parallel samples averaging ~3,000 tokens each, the total would be ~192,000 tokens (though the authors plot accuracy against the *average* thinking time per sample, as stated in the AIME24 plot labels). For REBASE (Figure 7), the authors explicitly note that "average thinking tokens for REBASE do not account for the additional compute from the reward model," meaning the reported token counts underestimate true compute by omitting the cost of running the 34B PRM at each step. For rejection sampling (Table 3, Figure 6), the budget is the maximum allowable thinking tokens per sample, with the actual compute cost including all rejected samples — the paper reports that achieving ≤3,500 token limit required an average of 655 attempts per sample, while ≤16,000 tokens required only ~1 attempt (Figure 6 caption). Budget forcing itself has zero additional compute cost beyond the tokens it causes the model to generate — the system only performs token-level string matching to detect the end-of-thinking delimiter and string concatenation to append "Wait" or the delimiter, which is negligible compared to model forward passes. For different evaluation runs in the test-time scaling curves, the x-axis values are chosen to span a meaningful range: for AIME24, evaluation runs at 1,024, 2,048, 4,096, and 8,192 thinking tokens (achieved by varying budget forcing parameters); for MATH500, runs at 512 and 2,048 tokens; for GPQA Diamond, runs at 1,024 and 4,096 tokens (Figure 1).

- **Cross-validation / statistical protocol.** The paper does not employ k-fold cross-validation for model selection — there is a single training run on the fixed s1K dataset. For the data ablation experiments (Table 2), the authors report **95% paired bootstrap confidence intervals** for differences relative to the s1K model, using 10,000 bootstrap samples. For example, when comparing 59K-full to s1K on AIME24, the confidence interval is [-13.3%, +20.0%], meaning "with 95% confidence, the true difference between 59K-full and s1K is between -13% and +20%." If the entire interval is negative (as with 1K-random: [-26.7%, -3.3%] on AIME24), the authors conclude performance is "confidently worse than s1K." For the test-time scaling method comparisons (Table 3), the number of evaluation runs |𝒜| is reported to indicate the robustness of the metric estimates — budget forcing uses 5 runs, rejection sampling uses 5, class-conditional control uses only 2. No formal statistical test (t-test, ANOVA) is reported for comparing scaling methods. The evaluation determinism concerns are documented in Appendix B: the authors note that "even when using the same random seeds and greedy sampling, evaluation scores can change significantly across runs" due to vLLM issues with different batch sizes, continuation handling, and tensor parallelism. They mitigate this by running final evaluations using full precision unless otherwise indicated, and by observing that "many generations are exactly the same for thousands of tokens and then suddenly differ in one token eventually ending up with an entirely different answer."

---

### Main Quantitative Results

#### Test-Time Scaling Behavior of s1-32B

The central result is that s1-32B equipped with budget forcing exhibits test-time scaling — accuracy improves with additional thinking tokens — across all three benchmarks. Figure 1 displays this:

- **MATH500:** Accuracy rises from ~65% at 512 thinking tokens to ~85% at 2,048 thinking tokens. The curve appears concave downward, suggesting diminishing returns within this range.
- **AIME24:** Accuracy rises from ~20% at 1,024 tokens to ~40% at 4,096 tokens, reaching 57% at 7,320 tokens (the rightmost point, corresponding to 6 forced "Wait" interventions). This represents an extrapolation from the baseline accuracy of 50.0% without budget forcing (Table 1, "s1 w/o BF").
- **GPQA Diamond:** Accuracy rises from ~40% at 1,024 tokens to ~60% at 4,096 tokens.

Figure 4(a) expands the AIME24 results to show the detailed budget forcing protocol. The model achieves approximately 50% accuracy with no forced waiting (the leftmost point, equivalent to "s1 w/o BF" in Table 1). Forcing a maximum of 2,048 or 4,096 thinking tokens produces points along the rising slope. The three rightmost points correspond to preventing the model from stopping its thinking 2, 4, and 6 times respectively, each time appending "Wait" to the current reasoning trace. At 6 interventions, accuracy reaches 56.7% (Table 3, Budget Forcing Performance). The paper notes that scaling "does eventually flatten out at six times" as the model enters repetitive loops rather than productive reasoning.

#### Sequential vs. Parallel Scaling

Figure 4(b) compares sequential scaling (budget forcing with s1-32B) against parallel scaling (majority voting with the base Qwen2.5-32B-Instruct). The base model is evaluated with 64 parallel samples at temperature 1, and majority voting is applied across 2, 4, 8, 16, 32, and 64 of these samples. The sequential scaling curve sits substantially above the parallel scaling curve across the entire range of test-time compute, demonstrating that "sequential scaling is more effective than parallel" (Section 4.2). The paper quantifies this as "validat[ing] our intuition from §3 that sequential scaling is more effective than parallel." The figure shows that even with 64-way majority voting, the base model cannot catch up to s1-32B's sequential scaling performance.

#### Benchmark Performance Against Other Models

Table 1 compares s1-32B against API-only and open-weight models on the three benchmarks:

| Model | AIME24 | MATH500 | GPQA Diamond |
|-------|--------|---------|--------------|
| o1-preview | 44.6 | 85.5 | 73.3 |
| o1-mini | 70.0 | 90.0 | 60.0 |
| o1 | 74.4 | 94.8 | 77.3 |
| Gemini 2.0 Flash Thinking | 60.0 | N.A. | N.A. |
| Qwen2.5-32B-Instruct (base) | 26.7 | 84.0 | 49.0 |
| QwQ-32B | 50.0 | 90.6 | 54.5 |
| r1 | 79.8 | 97.3 | 71.5 |
| r1-distill (Qwen-32B) | 72.6 | 94.3 | 62.1 |
| Sky-T1 | 43.3 | 82.4 | 56.8 |
| Bespoke-32B | 63.3 | 93.0 | 58.1 |
| s1 w/o BF | 50.0 | 92.6 | 56.6 |
| **s1-32B** | **56.7** | **93.0** | **59.6** |

Key comparisons: s1-32B at 56.7% on AIME24 exceeds o1-preview's 44.6% by 12.1 percentage points (a ~27% relative improvement, as claimed in the abstract: "exceeds o1-preview on competition math questions by up to 27%"). It matches o1-mini on neither AIME24 (70.0% vs. 56.7%) nor MATH500 (90.0% vs. 93.0% — here s1-32B is 3 points higher) nor GPQA (60.0% vs. 59.6% — essentially tied). It substantially trails o1 (74.4% vs. 56.7% on AIME24). Among open-weight models with open data, s1-32B achieves the highest AIME24 (56.7% vs. Bespoke-32B's 63.3% — wait, Bespoke-32B is actually higher at 63.3%, so this claim requires care; the paper says s1-32B is "the most sample-efficient open data reasoning model" in Figure 2, which is a different claim than "highest performing"). On MATH500, s1-32B ties Bespoke-32B at 93.0% and exceeds Sky-T1 (82.4%). On GPQA Diamond, s1-32B's 59.6% exceeds Sky-T1 (56.8%) and Bespoke-32B (58.1%) by small margins.

The sample-efficiency claim in Figure 2 (right) is visualized as a scatter plot with "Number of Examples" on a log-scale x-axis (from ~800 to ~800,000) and MATH500 Accuracy on the y-axis (80–100%). s1-32B sits at the top-left of the frontier: 1,000 examples, 93.0% MATH500 accuracy. r1-distill achieves 94.3% but with 800K examples (800× more). QwQ achieves 90.6% with unknown data quantity. Sky-T1 achieves 82.4% with 17K examples. Bespoke-Stratos achieves 93.0% with 17K examples — tying s1-32B's accuracy but requiring 17× more training data. o1-preview achieves 85.5% with unknown data. The frontier drawn in Figure 2 positions s1-32B as Pareto-optimal: no other model achieves higher accuracy with fewer training examples.

#### Sample Efficiency: s1K vs. Larger Datasets

Table 2 presents the core data ablation, comparing models trained on different 1K subsets and the full 59K dataset. All models are evaluated with budget forcing a maximum of ~30,000 thinking tokens (which "performs slightly better than the scores without BF (Table 1) as it allows the model to finish with a best guess when stuck in an infinite loop"). The s1K model achieves 50.0% AIME24 / 93.0% MATH500 / 57.6% GPQA Diamond in this configuration.

**1K-random** (random selection from quality-filtered pool, no difficulty/diversity filtering):
- AIME24: 36.7% — 13.3 percentage points lower than s1K, with 95% bootstrap CI [-26.7%, -3.3%] (entirely negative → confidently worse)
- MATH500: 90.6% — 2.4 points lower, CI [-4.8%, 0.0%] (barely excludes zero → marginally significant)
- GPQA: 52.0% — 5.6 points lower, CI [-12.6%, 2.5%] (includes zero → not statistically significant)

**1K-diverse** (uniform sampling across domains, no difficulty weighting):
- AIME24: 26.7% — 23.3 points lower, CI [-40.0%, -10.0%] (entirely negative → confidently worse)
- MATH500: 91.2% — 1.8 points lower, CI [-4.0%, 0.2%] (includes zero → not significant)
- GPQA: 54.6% — 3.0 points lower, CI [-10.1%, 5.1%] (includes zero → not significant)

**1K-longest** (selecting the 1,000 samples with longest reasoning traces, no diversity constraint):
- AIME24: 33.3% — 16.7 points lower, CI [-36.7%, 0.0%] (boundary includes zero → marginally significant)
- MATH500: 90.4% — 2.6 points lower, CI [-5.0%, -0.2%] (entirely negative → confidently worse)
- GPQA: 59.6% — 2.0 points *higher*, CI [-5.1%, 10.1%] (includes zero → not significant, though directionally positive)

**59K-full** (training on all 59,029 samples):
- AIME24: 53.3% — 3.3 points higher, CI [-13.3%, 20.0%] (wide interval spanning zero → not significant)
- MATH500: 92.8% — 0.2 points lower, CI [-2.6%, 2.2%] (includes zero → not significant)
- GPQA: 58.1% — 0.5 points higher, CI [-6.6%, 8.6%] (includes zero → not significant)

The key takeaway: 59K-full does not outperform s1K by a statistically significant margin on any benchmark, despite requiring 56× more training compute (394 vs. 7 H100 GPU hours). This demonstrates that data *selection* dominates data *quantity* for reasoning fine-tuning.

#### Test-Time Scaling Method Comparison

Table 3 compares budget forcing against alternative methods for controlling test-time compute, all evaluated on AIME24:

| Method | Control | Scaling | Performance | \|𝒜\| |
|--------|---------|---------|-------------|------|
| Budget Forcing (BF) | 100% | 15 | 56.7 | 5 |
| Token-Conditional Control (TCC) | 40% | -24 | 40.0 | 5 |
| TCC + BF | 100% | 13 | 40.0 | 5 |
| Step-Conditional Control (SCC) | 60% | 3 | 36.7 | 5 |
| SCC + BF | 100% | 6 | 36.7 | 5 |
| Class-Conditional Control (CCC) | 50% | 25 | 36.7 | 2 |
| Rejection Sampling (RS) | 100% | -35 | 40.0 | 5 |

Budget forcing achieves the best Performance (56.7%), 100% Control (tied with methods that also use forcing), and a positive Scaling of 15 (second only to CCC's 25, but CCC uses only 2 evaluation runs and has 50% Control). Rejection sampling achieves the worst Scaling (-35) — the negative value indicating that accuracy *decreases* with more allowed thinking tokens. Figure 6 visualizes this inverse relationship: accuracy drops from ~42% at 3,072 average thinking tokens to ~32% at 5,120 tokens.

#### Rejection Sampling Inverse Scaling

Figure 6 plots AIME24 accuracy against average thinking tokens when rejection sampling with temperature 1. The five data points (from left to right) correspond to sampling until all generations have less than 3,500, 4,000, 5,000, 8,000, and 16,000 thinking tokens, requiring an average of 655, 97, 8, 3, 2, and 1 tries per sample respectively. The curve slopes downward: ~42% at the strictest budget (3,500 tokens) to ~32% at the loosest (16,000 tokens). This is the "inverse scaling trend" the authors highlight. The mechanistic explanation (Section 5.2): "shorter generations tend to be the ones where the model was on the right track from the start, whereas longer ones tend to be ones where the model made mistakes and thus backtracks or questions itself." The qualitative example in §E.2 (Table 15) demonstrates this concretely: the same AIME problem, when rejection-sampled for ≤4,000 thinking tokens, produces a correct answer through a direct, correct approach; when sampled for ≤8,000 tokens, the same model backtracks extensively, gets confused, and converges on an incorrect answer.

#### Budget Forcing Extrapolation Ablations

Table 4 compares different strings appended when suppressing the end-of-thinking delimiter twice (2x):

| Model | AIME24 | MATH500 | GPQA Diamond |
|-------|--------|---------|--------------|
| No extrapolation (baseline) | 50.0 | 93.0 | 57.6 |
| 2x without string | 50.0 | 90.2 | 55.1 |
| 2x "Alternatively" | 50.0 | 92.2 | 59.6 |
| 2x "Hmm" | 50.0 | 93.0 | 59.6 |
| 2x "Wait" | 53.3 | 93.0 | 59.6 |

"Wait" is the only string that improves AIME24 performance (from 50.0% to 53.3%, a 3.3 percentage point gain). "Hmm" and "Alternatively" match the baseline on AIME24 (50.0%) while improving GPQA (59.6% vs. 57.6% baseline). Appending no string degrades MATH500 (90.2% vs. 93.0%) and GPQA (55.1% vs. 57.6%). The authors conclude that "Wait" generally gives the best performance, though the mechanism (whether it triggers self-correction behavior specifically, or simply provides a more natural continuation) is not isolated.

#### Parallel Scaling as a Complement to Sequential Scaling

Figure 7 compares three scaling methods on AIME24: **sequential scaling** (prompting the model to use up to 32, 64, 256, and 512 steps), **REBASE** tree search (with 16 parallel trajectories and majority voting aggregation), and **majority voting** (also with 16 parallel trajectories). The results show:

- **Sequential scaling** achieves the highest accuracy at the leftmost point (~30% at ~2,048 tokens) but drops sharply at the rightmost point (from ~50% to ~35% at ~130K tokens) because "for 12 out of the 30 evaluation questions the model generates a response that exceeds the context window leading to a large performance drop."
- **REBASE** scales consistently from ~30% at 2,048 tokens to ~50% at 130K tokens, surpassing sequential scaling at higher budgets. However, "REBASE requires an additional forward pass at each step for the reward model adding some computation overhead" — meaning the true compute exceeds the reported thinking tokens.
- **Majority voting** scales from ~22% at 2,048 tokens to ~38% at 130K tokens, consistently below both REBASE and sequential scaling.

The paper interprets this as evidence that "parallel scaling methods complement sequential scaling thus they offer an avenue for scaling test-time compute even further; beyond fixed context windows."

---

### Ablation Studies and Robustness Checks

**Data selection criteria (Quality, Difficulty, Diversity):** Table 2 demonstrates that all three criteria jointly are necessary. The ablation structure isolates each: 1K-random (only Quality, no Difficulty/Diversity) degrades AIME24 by 13.3 points (CI [-26.7%, -3.3%]); 1K-diverse (Quality + Diversity, no Difficulty weighting) degrades AIME24 by 23.3 points (CI [-40.0%, -10.0%]); 1K-longest (Quality + Difficulty via length, no Diversity constraint) degrades AIME24 by 16.7 points (CI [-36.7%, 0.0%]). The non-significance of the 59K-full difference (CI [-13.3%, +20.0%] on AIME24) establishes that data selection, not total data volume, is the primary driver of performance — a finding that holds conditional on the specific selection algorithm used for s1K.

**Training sequence length:** Table 8 compares training with a short sequence length (4,096 tokens, cutting off 74% of training samples) against a long sequence length (32,768 tokens, cutting off 0%). The short-sequence model generates substantially longer reasoning traces at test time (20,721 vs. 6,984 average thinking tokens on AIME24) but achieves worse accuracy (30.0% vs. 50.0% on AIME24, 90.0% vs. 91.0% on MATH500, 52.5% vs. 53.0% on GPQA). The mechanism is that truncated training samples lack the answer section, so the model receives weaker gradients for the thinking → answer transition, leading to longer, less efficient reasoning. This is a non-obvious finding: longer test-time reasoning is not always better, and training data completeness (ensuring all samples include the answer transition) is crucial for efficient test-time behavior.

**Budget forcing string choice:** Table 4 ablates the appended string (none, "Alternatively", "Hmm", "Wait") when extending thinking 2x. Only "Wait" improves AIME24 (53.3% vs. 50.0% baseline). "Hmm" and "Wait" both improve GPQA (59.6% vs. 57.6%). No string degrades MATH500 (90.2% vs. 93.0%). The finding is that the specific string matters, and "Wait" provides the most consistent gains, though the paper does not isolate whether this is due to semantic priming (encouraging self-correction) or distributional matching (being a natural continuation in reasoning text).

**Class-conditional control prompts:** Table 14 compares two generic prompts appended to the question: "Answer after a short amount of thinking" vs. "Answer after a long amount of thinking." The long-thinking prompt increases thinking tokens (9,651 vs. 8,033 on AIME24) and improves AIME24 accuracy (36.7% vs. 30.0%), but degrades GPQA (51.0% vs. 56.6%) and underperforms the baseline without any prompt (50.0% / 93.0% / 57.6% on AIME24 / MATH500 / GPQA). The conclusion is that class-conditional control provides some scaling but is unreliable and imprecise.

**Step-conditional control compensation effects:** Table 13 shows that when forced to use fewer steps (16 vs. 256), the model compensates by making each step longer (96 vs. 56 tokens per step), keeping total thinking tokens roughly constant (7,252 vs. 7,551, a difference of only ~300 tokens despite a 16× difference in step budget). This demonstrates a fundamental limitation: the model "hacks its way around the compute constraint" by varying tokens per step to maintain its preferred total reasoning length. Combining step-conditional control with budget forcing solves the control issue (100% Control in Table 3) but still underperforms pure budget forcing (36.7% vs. 56.7% Performance), likely because the step-counting overhead adds tokens without contributing to reasoning quality.

**Token-conditional control failure:** Table 12 shows that the model does not follow token-length instructions: when told to think for 1,024 tokens, it generates 7,939; when told 16,384, it generates 7,500. The generation length is largely insensitive to the instruction. Adding budget forcing (capping when limit is reached) achieves perfect control but Performance remains at 40.0% (Table 3, TCC + BF), well below pure budget forcing's 56.7%. This suggests that training with token instructions actually harms the model's reasoning quality, possibly because the instruction in the prompt interferes with natural reasoning.

**REBASE parallel scaling augmentation:** Figure 7 shows that REBASE tree search with majority voting scales from ~30% to ~50% AIME24 accuracy as thinking tokens increase from ~2,048 to ~131,072, outperforming majority voting (which reaches ~38% at the same budget). However, the paper notes two caveats: (1) REBASE requires an additional forward pass through a 34B reward model at each step, so the true compute exceeds reported thinking tokens; (2) the comparison is only for 16 parallel trajectories, so scaling to larger parallel budgets is not tested. Even so, REBASE provides the best scaling at high token budgets, suggesting sequential + parallel hybrid approaches could push performance further.

**Gemini distillation quality:** The paper uses Gemini Flash Thinking to generate reasoning traces, with 53.6% of s1K traces deemed correct by Claude 3.5 Sonnet (§2.2). The follow-up s1K-1.1 (Appendix A) regenerates traces using DeepSeek R1, achieving 63.0% correctness, and s1.1 shows substantially improved performance (Table 5: 56.7% vs. 50.0% on AIME24 without budget forcing, 56.7% vs. 53.3% with 2x "Wait"). This demonstrates that distillation quality matters — better source models produce better training data — but also that even imperfect traces (47.4% incorrect in s1K) are sufficient to teach the desired reasoning format.

**Evaluation determinism issues (Appendix B):** The authors document significant variability in evaluation scores due to vLLM implementation details: different batch sizes, continuation handling, and tensor parallelism can cause "many generations that are exactly the same for thousands of tokens and then suddenly differ in one token eventually ending up with an entirely different answer." They mitigate this by running final evaluations in full precision. This is not an ablation but a methodological concern that could affect reproducibility of exact numbers, though the paper argues the relative comparisons (s1K vs. ablations, budget forcing vs. baselines) remain valid since all models are evaluated under the same conditions.

---

### Critical Assessment

**Claim 1: "Training on only 1,000 samples leads to a strong reasoning model that scales in performance with more test-time compute."** This claim is supported by Figure 1, Figure 4(a), and Table 1. The test-time scaling curves are unambiguous: accuracy on all three benchmarks improves as thinking tokens increase. However, the claim requires careful qualification regarding what "scales" means. The AIME24 scaling from 50.0% (no BF) to 56.7% (6 "Wait" interventions) is a meaningful 6.7 percentage point gain, but it flattens at 6 interventions — the paper explicitly states it "does eventually flatten out at six times." The scaling is therefore demonstrated over a specific, limited range (roughly 1,000–8,000 tokens), not unlimited extrapolation. The MATH500 and GPQA curves in Figure 1 show only 2–3 data points each, making the scaling claim on those benchmarks based on sparse sampling. A more thorough investigation would have evaluated at intermediate budgets (e.g., 4 points on MATH500 and GPQA, as was done for the rejection sampling curve in Figure 6) to characterize the functional form of scaling.

A deeper weakness: the scaling behavior is demonstrated only on s1-32B, a single model trained on a single dataset. The paper does not show whether the same budget forcing technique produces scaling behavior on the base Qwen2.5-32B-Instruct without fine-tuning, or on other models fine-tuned on s1K (e.g., smaller Qwen variants). This makes it impossible to attribute the scaling behavior specifically to the s1K fine-tuning — it's possible that budget forcing would produce scaling on *any* instruction-tuned model, or that it's specific to the Qwen family. The parallel scaling comparison in Figure 4(b) shows majority voting with the base model, but does not show budget forcing applied to the base model. An ablation: "does budget forcing work on Qwen2.5-32B-Instruct without s1K fine-tuning?" would isolate whether the fine-tuning enables scaling or merely raises the baseline.

**Claim 2: "s1-32B exceeds o1-preview on competition math questions by up to 27%."** The "up to 27%" figure comes from comparing s1-32B's AIME24 score (56.7%) to o1-preview's (44.6%), representing a relative improvement of (56.7 - 44.6) / 44.6 ≈ 27.1%. This claim is factually supported by Table 1. However, it's selective in two ways: (1) "up to" is doing work — on MATH500, s1-32B's 93.0% exceeds o1-preview's 85.5% by only 8.8% relative, and on GPQA Diamond, s1-32B's 59.6% trails o1-preview's 73.3% substantially (a 18.7% relative *decrease*). The paper does not claim s1-32B exceeds o1-preview on GPQA — the abstract specifically says "competition math questions," which is accurate. (2) The comparison is specifically to o1-preview, not to o1-mini (70.0% on AIME24) or o1 (74.4%) — both of which substantially outperform s1-32B. This is not hidden; Table 1 clearly shows the full o1 series, and the abstract specifies o1-preview. But readers should understand that "exceeds o1-preview" is the ceiling of what s1-32B achieves relative to the o1 family, not a general claim of parity.

The more fundamental question is what this comparison actually demonstrates. o1-preview was released in September 2024 and represents OpenAI's earlier reasoning effort; o1 and o3-mini (released later) show substantially higher performance. If the goal is to demonstrate that "simple methods can achieve competitive reasoning," then beating an older, weaker model in the same family is less compelling than matching current open-weight models. Table 1 shows that r1-distill (72.6% on AIME24) and Bespoke-32B (63.3%) both outperform s1-32B, though they use 800× and 17× more training data respectively. The paper's framing around sample efficiency rather than absolute performance is appropriate, but the o1-preview comparison in the abstract may overstate the practical competitiveness of the approach.

**Claim 3: "Combining all three criteria — Quality, Difficulty, Diversity — via our methodology is key for sample-efficient reasoning training."** Table 2 strongly supports this claim: each ablation that removes one criterion produces statistically significant degradation on at least one benchmark. However, the claim is established only for the specific s1K selection algorithm, not for the general principle. Would a different difficulty filter (e.g., using only Qwen2.5-7B-Instruct instead of both 7B and 32B) work? Would a different diversity criterion (e.g., embedding-based clustering rather than MSC domain classification) work? Would a different difficulty proxy (e.g., using predicted vs. actual reasoning length instead of Gemini trace length) work? The paper demonstrates that *this specific combination* of criteria works, but does not ablate variations within each criterion to establish robustness to implementation choices.

The confidence intervals in Table 2 also reveal an important caveat: the differences between s1K and the ablated datasets are statistically significant primarily on AIME24, not on MATH500 or GPQA. For 1K-random on MATH500, the CI is [-4.8%, 0.0%] — just barely excluding zero. For 1K-diverse on GPQA, the CI is [-10.1%, 5.1%] — wide and spanning zero. This means the data selection criteria matter most for the hardest benchmark (AIME24), and their effect on easier benchmarks (MATH500, GPQA) is less precisely measurable. This makes intuitive sense — careful selection of difficult, diverse problems matters most when the evaluation itself is difficult — but it qualifies the strength of the general claim.

**Claim 4: "Budget forcing leads to the best scaling as it has perfect controllability with a clear positive slope leading to strong performance."** Table 3 robustly supports this relative to the tested baselines: budget forcing dominates on Control (100%), is competitive on Scaling (15, second to CCC's 25 but CCC has only 50% Control and 2 evaluation runs), and achieves the best Performance (56.7%). However, the baseline set is limited in several ways:

1. **No comparison to process reward model (PRM) guided search.** The paper compares to REBASE in Figure 7, but REBASE is evaluated with majority voting aggregation, not as a standalone scaling method with its own Control/Scaling/Performance metrics. A direct comparison of budget forcing against PRM-guided beam search or best-of-N weighted selection (as in the reference paper analyzed earlier) would strengthen the claim that budget forcing is the "best" method.

2. **Only s1-32B is tested with budget forcing.** The paper does not apply budget forcing to other models (Qwen base, other fine-tuned variants) to test whether the technique generalizes or is specific to s1-32B's training. If budget forcing works because s1-32B was specifically trained with delimiters, that's an important scope limitation: the method requires models trained with explicit `<|im_start|>think` / `<|im_start|>answer` delimiters.

3. **The Scaling metric has high variance with small |𝒜|.** Budget forcing uses 5 evaluation runs; class-conditional control uses only 2. The Scaling metric averages slopes across all pairs, meaning with 5 runs there are `$\binom{5}{2} = 10$` pairs, and with 2 runs there is only 1 pair. CCC's Scaling of 25 is based on a single slope estimate, making it highly unreliable. The paper notes this by including |𝒜| in Table 3, but doesn't compute uncertainty on the Scaling estimates (which would be large for small |𝒜|).

4. **Rejection sampling's inverse scaling is well-demonstrated but the mechanism is not experimentally isolated.** The paper hypothesizes that "shorter generations tend to be the ones where the model was on the right track from the start," but this could be tested: one could measure the correlation between generation length and correctness within the unconstrained generation distribution, or examine whether forcing the model to stop early (via budget forcing) preserves the higher accuracy of short natural generations. The qualitative example in §E.2 is illustrative but not systematic.

**Missing experiments that would strengthen the paper:**

- **Budget forcing applied to the base (un-fine-tuned) model.** Does the base Qwen2.5-32B-Instruct show test-time scaling with budget forcing? If yes, what role does s1K fine-tuning play beyond raising the baseline? If no, what specific aspect of the fine-tuning enables scaling? This is the most important missing ablation for attributing the scaling behavior to the training procedure.

- **Intermediate budget points on MATH500 and GPQA.** Figure 1 shows only 2–3 points per benchmark. Four or five budget levels per benchmark would characterize the functional form of scaling (linear? logarithmic? sigmoidal?) and identify where flattening begins.

- **Comparison of budget forcing against best-of-N with s1-32B.** Figure 4(b) compares sequential (s1-32B with budget forcing) against parallel (base model with majority voting), but not against parallel scaling of s1-32B itself. Would majority voting or best-of-N with the fine-tuned model outperform budget forcing at high budgets? This is a natural baseline that is absent.

- **Statistical testing for test-time scaling method comparisons.** The raw numbers in Table 3 have no confidence intervals, making it impossible to assess whether budget forcing's Performance advantage (56.7 vs. 40.0 for RS) is statistically significant or could be noise from the 30-question AIME24 test set.

- **Difficulty-stratified analysis.** The paper does not break down results by question difficulty within each benchmark. As the reference paper demonstrated, test-time scaling behavior often varies dramatically by difficulty — beam search helps on medium problems but hurts on easy ones due to verifier over-optimization. Without difficulty stratification, the reported scaling curves may be averaging over qualitatively different behaviors. For AIME24 (30 questions), stratifying by difficulty might be underpowered, but for MATH500 (500 questions), it would be informative.

- **Sensitivity to the exact s1K composition.** The paper uses a deterministic algorithm to select s1K. Would a different random seed in the domain sampling step (Algorithm 1) produce a meaningfully different 1K subset with different performance? Bootstrapping the selection process and reporting variance would indicate how stable the s1K advantage is.

**Limitations that constrain the strength of conclusions:**

- **Single model family, single scale.** All experiments use Qwen2.5-32B-Instruct. It is unknown whether the findings generalize to other model families (Llama, Mistral), other scales (7B, 70B), or base models without instruction tuning. The authors argue the model is "representative," but this is an untested assumption.

- **Test sets are small for some claims.** AIME24 has only 30 questions. A single-question swing changes accuracy by 3.3 percentage points. The 6.7-point improvement from budget forcing (50.0% → 56.7%) represents only 2 additional correct answers out of 30. While the scaling trend is clear across multiple budget levels, the absolute difference is based on a small sample.

- **The "simple" approach still requires access to Gemini Flash Thinking API for distillation.** The paper's framing emphasizes simplicity, but the data pipeline depends on a proprietary, closed-source model (Gemini) to generate reasoning traces. This creates a dependency that limits full reproducibility — if Gemini's API changes or becomes unavailable, the exact s1K dataset cannot be recreated. The follow-up s1K-1.1 using DeepSeek R1 (Appendix A) partially mitigates this by showing that open-weight models can substitute, but the core s1K dataset is Gemini-dependent.

- **The compute cost of difficulty estimation is unaccounted for.** The difficulty filter evaluates two models (Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct) on 24,496 questions, with grading done by Claude 3.5 Sonnet. This is a substantial computational expense that is not amortized in any reported metric — it's a one-time data curation cost, not a per-question inference cost, but it means the total compute to produce s1-32B is substantially higher than the 7 H100 GPU hours of fine-tuning alone.

- **No latency analysis.** Budget forcing's sequential nature means that extending thinking by 6 "Wait" interventions adds serial dependencies — each intervention requires the model to reach the point where it wants to stop, which cannot be parallelized. The paper measures cost in thinking tokens but does not report wall-clock time. For latency-sensitive applications, 7,320 tokens of sequential generation may be impractical regardless of accuracy gains.

**Overall assessment:** The experiments convincingly demonstrate that s1-32B with budget forcing exhibits test-time scaling behavior on three benchmarks, that the s1K data selection criteria jointly matter for performance, and that budget forcing provides better controllability than alternative prompting-based methods. The sample-efficiency frontier in Figure 2 is a genuine contribution — s1-32B achieves competitive MATH500 accuracy with 1–2 orders of magnitude fewer training examples than comparably performing models. However, the claims about budget forcing being the "best" method are established only against a limited set of baselines that exclude RL-based approaches and PRM-guided search. The generalization of findings beyond Qwen2.5-32B-Instruct and beyond the specific s1K selection algorithm remains untested. The most significant missing experiment — applying budget forcing to the base model without s1K fine-tuning — would clarify whether the scaling behavior is a property of the fine-tuning or of the inference-time intervention itself.

## 6. Limitations and Trade-offs

### 6.1 Test-Time Scaling Flattens at a Modest Ceiling and Fails on the Hardest Problems

**The assumption or constraint.** Budget forcing extends reasoning by suppressing the end-of-thinking delimiter and appending `"Wait"`, but the paper explicitly acknowledges this does not scale indefinitely:

> "it eventually flattens out (Figure 4), and the context window of the underlying language model constrains it." (Section 6.2)

> "Suppressing the end-of-thinking token delimiter too often can lead the model into repetitive loops instead of continued reasoning." (Section 4.2)

The scaling ceiling is reached at roughly 6 forced "Wait" interventions, producing a maximum of 56.7% on AIME24 — a 6.7 percentage point improvement over the baseline 50.0% (Figure 4a, Table 1). Beyond this, the model either loops or exceeds the maximum context length (12 out of 30 AIME24 questions exceeded the context window when using step-conditional control with up to 512 steps, as noted in Section 6.2).

**The consequence.** This limitation means budget forcing provides a **bounded improvement** — it is useful for moderate test-time compute expansion (roughly 1–8K thinking tokens), but it cannot serve as a mechanism for arbitrary scaling. A practitioner hoping to "spend more compute to solve harder problems" will encounter a hard ceiling where additional interventions produce no further gains or actively degrade performance (looping). The method fundamentally cannot push performance beyond the model's ceiling as established by its training and architecture. This contrasts with parallel scaling methods like REBASE, which continue to improve at higher budgets (Figure 7: REBASE reaches ~50% on AIME24 at ~130K tokens, while sequential scaling drops to ~35% at the same budget due to context overflow). The flattening also means that budget forcing cannot substitute for more powerful base models on problems that require reasoning depth beyond what the 32B model can produce, even with extended thinking.

**What evidence exists in the paper.** Figure 4(a) shows the AIME24 scaling curve: the three rightmost data points correspond to 2, 4, and 6 "Wait" interventions, with the curve visibly plateauing. Table 3 quantifies the maximum Performance at 56.7% (5 evaluation runs). Figure 7 shows that at high token budgets (~130K), sequential scaling via step-conditional control drops from ~50% to ~35% because of context window overflow, while REBASE continues to scale. The quantitative gain from budget forcing — 50.0% → 56.7% on AIME24 — is a 6.7 percentage point absolute improvement, which on the 30-question AIME24 test set represents approximately 2 additional correct answers. This is a meaningful but modest absolute gain, and the method provides no pathway to the 70–80% range achieved by r1 or o1.

**Mitigation status.** The paper partially addresses this by proposing parallel scaling methods as complements. Figure 7 demonstrates that REBASE tree search scales further than sequential methods at high budgets, and the authors conclude:

> "parallel scaling methods complement sequential scaling thus they offer an avenue for scaling test-time compute even further; beyond fixed context windows."

However, this is a separate method (requiring a process reward model) rather than an extension of budget forcing itself. The paper does not propose or test any modifications to budget forcing that would extend its scaling range — such as rotating through different intervention strings, increasing temperature during forced continuations to escape loops, or combining sequential and parallel strategies adaptively. The fundamental ceiling — that the model eventually loops or exceeds its context window — is acknowledged as an open problem:

> "An exciting direction for future work is also researching whether applying budget forcing to a reasoning model trained with reinforcement learning yields better extrapolation; or if RL allows for new ways of test-time scaling beyond budget forcing."

This is a clear statement that budget forcing alone is insufficient for continued scaling, and that RL-trained models may have fundamentally different scaling properties.

### 6.2 The Difficulty Estimation and Data Curation Pipeline is Computationally Expensive but Unaccounted

**The assumption or constraint.** The s1K data selection algorithm (Section 2.2) requires evaluating two models (Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct) on 24,496 candidate questions, with correctness assessed by Claude 3.5 Sonnet comparing each attempt against a Gemini-generated reference solution. This is the **difficulty filter** — removing questions that either model can solve correctly. Additionally, the initial distillation step generates reasoning traces for all 59,029 questions using the Gemini Flash Thinking API. The paper reports the fine-tuning cost (26 minutes on 16 H100 GPUs, or ~7 H100 GPU hours), but **does not account for the cost of producing the training data** in any headline metric. The full 59K dataset requires 394 H100 GPU hours to train on (Table 2), but the cost of *creating* those 59K traces via Gemini API calls is not quantified.

**The consequence.** The paper's framing of "simplicity" and "sample efficiency" refers narrowly to the fine-tuning step. A practitioner seeking to replicate the approach from scratch would need: (1) API access to Gemini Flash Thinking (or a comparably capable model) to generate 59K reasoning traces; (2) API access to Claude 3.5 Sonnet to grade 24,496 × 2 = 48,992 model attempts for difficulty filtering; (3) compute to run Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct on 24,496 questions. These costs likely dominate the fine-tuning cost, but they are invisible in the "26 minutes on 16 H100 GPUs" headline. This matters for two reasons: first, it makes the true cost of the approach substantially higher than advertised; second, it creates a dependency on proprietary APIs (Gemini, Claude) that contradicts the paper's open-source values. If either API becomes unavailable, changes pricing, or alters model behavior, the exact s1K dataset cannot be reproduced.

**What evidence exists in the paper.** The paper does not quantify the API costs or compute requirements for data generation. Table 2 compares fine-tuning costs (7 vs. 394 H100 GPU hours for s1K vs. 59K-full) but omits data creation costs entirely. Section 2.2 describes the difficulty filter: "We evaluate two models on each question: Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct, with correctness assessed by Claude 3.5 Sonnet." The number of model evaluations (48,992) and grading calls is not stated explicitly, but follows from the pipeline: 24,496 questions × 2 models = 48,992 forward passes. The initial 59K Gemini API calls for trace generation are described in Section 2.1: "For each question, we generate a reasoning trace and solution using the Google Gemini Flash Thinking API."

**Mitigation status.** The paper does not attempt to mitigate this limitation. There is no analysis of cheaper difficulty estimation methods (e.g., using only one model, using a smaller model, skipping difficulty filtering entirely and relying only on trace length as in "1K-longest"). The 1K-longest ablation in Table 2 (33.3% on AIME24 vs. 50.0% for s1K) shows that skipping the difficulty filter degrades performance substantially, so the cost appears necessary for the reported results. The follow-up s1K-1.1 (Appendix A) regenerates traces using DeepSeek R1 rather than Gemini, which mitigates the API dependency for the trace generation step (R1 is open-weight) but not for the grading step (Claude 3.5 Sonnet for correctness assessment, Claude 3.7 for final grading). The paper does not discuss the feasibility of using open-weight models for grading, which would close the reproducibility gap.

### 6.3 Findings are Demonstrated on a Single Model Family at a Single Scale

**The assumption or constraint.** All experiments — data curation, fine-tuning, budget forcing, ablations — use Qwen2.5-32B-Instruct (Qwen et al., 2024) as the base model. The paper justifies this choice by stating that the model "is representative of the capabilities of many contemporary LLMs" and "on math tasks generally matches or outperforms the larger Qwen2.5-72B-Instruct" (Section 4.1). However, no experiments are conducted with other model families (Llama, Mistral, DeepSeek), other scales (7B, 70B), or base models without instruction tuning. The budget forcing technique is tested only on s1-32B itself; it is never applied to the un-fine-tuned Qwen2.5-32B-Instruct to test whether the scaling behavior is a product of the fine-tuning or the inference intervention.

**The consequence.** A practitioner considering this approach for a different model family (e.g., a Llama-based system) has no evidence that the findings transfer. The "Superficial Alignment Hypothesis" — that pretrained models already possess reasoning capabilities that fine-tuning merely activates — implies that the approach *should* generalize, since the capability is latent in pretraining rather than specific to Qwen. But this is untested. Several aspects of the approach could be model-specific: (1) Qwen2.5-32B-Instruct's base pass@1 on MATH500 is 84.0% (Table 1), meaning it already has strong math capability — a model with weaker base math performance might not show the same gains from 1K reasoning examples; (2) the `<|im_start|>think` / `<|im_start|>answer` delimiter format is Qwen-specific and would need adaptation for models with different chat templates; (3) the optimal budget forcing string ("Wait") might differ across model families depending on their pretraining data distribution.

**What evidence exists in the paper.** No cross-model experiments are reported. Table 1 benchmarks s1-32B against other model families (o1, r1, QwQ, Sky-T1, Bespoke) but does not apply the s1K + budget forcing recipe to those models. Figure 4(b) compares sequential scaling (s1-32B with budget forcing) against parallel scaling (Qwen2.5-32B-Instruct with majority voting), but does not apply budget forcing to the un-fine-tuned model — this would directly test whether the fine-tuning is necessary for scaling behavior. Section 6.1 states the hypothesis:

> "We hypothesize that the model is already exposed to large amounts of reasoning data during pretraining which spans trillions of tokens. Thus, the ability to perform reasoning is already present in our model. Our sample-efficient finetuning stage just activates it."

This hypothesis is plausible but unverified across model families. If the hypothesis is correct, budget forcing should produce scaling behavior on *any* sufficiently capable base model after minimal SFT; if incorrect, the approach may be specific to Qwen2.5's particular pretraining or instruction tuning recipe.

**Mitigation status.** The paper does not address this limitation beyond stating the belief in representativeness. The follow-up s1.1 (Appendix A) uses the same Qwen2.5-32B-Instruct base model with improved training data (traces from DeepSeek R1 rather than Gemini), so it also does not test cross-model generalization. Future work would need to replicate the s1K selection + budget forcing pipeline on models like Llama-3-70B-Instruct or Mistral-Large to establish generality. Until then, the findings should be understood as demonstrated for Qwen2.5-32B-Instruct specifically, with theoretical but unverified generalization to other models.

### 6.4 Budget Forcing is Purely Sequential and Introduces Latency Bottlenecks

**The assumption or constraint.** Budget forcing operates sequentially: each forced "Wait" intervention requires the model to generate until it attempts to produce the end-of-thinking delimiter, at which point the system intervenes, appends `"Wait"`, and continues generation. This process cannot be parallelized — the model must serially generate reasoning, reach a stopping point, be modified, and continue. The paper measures cost exclusively in **thinking tokens** (a proxy for total FLOPs) and does not report wall-clock time or latency for any experiment.

**The consequence.** For latency-sensitive applications — interactive assistants, real-time decision-making, user-facing chatbots — the 6 sequential "Wait" interventions that produce the best AIME24 performance (56.7%) may be impractical. If each reasoning segment between interventions requires, say, 1,000 tokens of generation, the total generation involves 6 sequential blocks that cannot be computed in parallel. In contrast, parallel scaling methods like majority voting or best-of-N can run all N samples simultaneously given sufficient hardware, trading throughput for latency. A practitioner deciding between sequential budget forcing (6× serial chain, ~7,000 token total) and parallel best-of-64 (64× parallel samples, each ~3,000 tokens, total ~192,000 tokens but wall-clock time of only ~3,000 tokens worth of generation) would need latency numbers that the paper does not provide.

**What evidence exists in the paper.** No latency measurements are reported. The AIME24 scaling curve in Figure 4(a) uses "Average thinking time (tokens)" as the x-axis, which is a compute measure but not a time measure. The paper mentions that "budget forcing provides perfect control, good scaling, and leads to our best AIME24 score" (Section 5.2), but control refers to token budgets, not time budgets. The context window limitation discussed in Section 6.2 — "for 12 out of the 30 evaluation questions the model generates a response that exceeds the context window" — is an indirect latency signal, since very long generations take proportionally longer, but the paper does not quantify this.

**Mitigation status.** Not addressed. The paper's evaluation metrics (Section 3.2) — Control, Scaling, Performance — do not include latency or throughput. This is a deliberate scope choice (the paper focuses on "test-time compute" as a FLOPs concept), but it means the metrics are incomplete for practical deployment decisions. The paper's suggestion that parallel methods "complement sequential scaling" (Section 6.2) points toward hybrid approaches that could balance latency and compute, but no such hybrid (e.g., parallel chains each with sequential budget forcing) is tested. A latency-aware evaluation would reveal whether the sequential nature of budget forcing makes it suitable only for offline/batch settings where wall-clock time is not a constraint.

### 6.5 All Evaluations are on Mathematics and Science Benchmarks with Clean Verifiable Answers

**The assumption or constraint.** The three benchmarks — AIME24, MATH500, GPQA Diamond — all consist of problems with clean, verifiable ground-truth answers (integer answers 000–999 for AIME24, specific mathematical expressions or multiple-choice options for MATH500 and GPQA). The paper's data curation pipeline similarly relies on clean correctness signals: the difficulty filter grades model attempts using Claude 3.5 Sonnet comparing against a reference solution, and the data selection algorithm favors problems where Gemini 2.0 Flash Thinking generated a correct solution (for AIME/GPQA) or a long reasoning trace (for MATH500). No experiments are conducted on open-ended generation tasks, creative writing, dialogue, summarization, code generation (beyond the initial 59K data collection, which included LiveCodeBench and USACO), or any domain where "correctness" is subjective or multi-dimensional.

**The consequence.** The approach may not generalize to domains where difficulty estimation and correctness verification are harder. The entire s1K selection algorithm depends on: (1) being able to automatically grade model attempts to filter easy questions (requiring a verifier, which in this case is Claude 3.5 Sonnet prompted with a reference solution); (2) being able to measure reasoning trace length as a difficulty proxy (which assumes harder problems require longer reasoning, a plausible but domain-specific assumption); (3) having a taxonomy of domains (the Mathematics Subject Classification) to ensure diversity. For open-ended tasks — writing a persuasive essay, generating a creative story, answering an ambiguous philosophical question — none of these components transfer straightforwardly. Difficulty estimation would need a different proxy (e.g., human preference scores), diversity classification would need a different taxonomy, and trace length may not correlate with quality. The budget forcing technique itself — appending `"Wait"` to trigger self-correction — may work differently (or not at all) in domains where problems do not have a single correct answer that can be "double-checked."

**What evidence exists in the paper.** All reported results are on math and science benchmarks. The paper lists broader data sources in Table 7 (including legal reasoning, crossword puzzles, logic), but these are only in the initial 59K pool, not in the final trained model's evaluation. Figure 1 shows scaling curves only for MATH500, AIME24, and GPQA Diamond. The examples in Figure 5 are all from math and physics. There is no evaluation on code generation (HumanEval, MBPP), multi-step planning, factuality, or any benchmark that does not reduce to a single correct answer.

**Mitigation status.** Not addressed. The paper does not claim generality beyond reasoning tasks, and the title "Simple test-time scaling" is qualified by the benchmarks used. However, the abstract's claim that "our model s1-32B exceeds o1-preview on competition math questions" implicitly limits the scope to math, which is appropriate. A practitioner interested in non-math reasoning domains would need to independently validate whether the s1K selection criteria (difficulty via model failure, diversity via MSC classification, quality via format checking) and budget forcing transfer.

### 6.6 Evaluation Data Leakage Risk from Training on Distilled Traces of Models That Were Trained on Similar Data

**The assumption or constraint.** The reasoning traces in s1K are distilled from Gemini Flash Thinking Experimental (and in s1K-1.1, from DeepSeek R1). These source models were themselves trained on large-scale web data that may overlap with the evaluation benchmarks (MATH500, AIME24, GPQA Diamond). The authors perform decontamination using 8-gram overlap between training questions and evaluation questions (Section 2.1, Appendix C.5):

> "We decontaminate all samples against our evaluation questions (MATH500, GPQA Diamond, AIME24) using 8-grams and deduplicate the data."

However, this only filters questions with near-exact textual overlap with evaluation questions, not more subtle forms of leakage. The reasoning *traces* generated by Gemini for training questions may contain solution patterns or reasoning templates that transfer to evaluation questions through conceptual similarity, even if the questions themselves do not share 8-grams.

**The consequence.** There is a risk that s1-32B's performance partly reflects distillation from Gemini's knowledge of the evaluation benchmarks — knowledge that may have been acquired through training on benchmark data or similar problems — rather than genuine reasoning capability. If Gemini was trained on MATH500 problems or their solutions, its reasoning traces for similar training questions might encode correct solution strategies that s1-32B then memorizes through SFT, inflating its apparent reasoning ability. The 8-gram decontamination addresses only literal overlap at the question level, not conceptual leakage at the reasoning level. This is a well-known challenge in benchmark-based evaluation: models trained on distilled traces from frontier models may inherit those models' benchmark contamination rather than acquiring generalizable reasoning skills. The paper's claim of "sample-efficient reasoning" would be weakened if the sample efficiency reflects distillation from a contaminated source rather than genuine capability activation.

**What evidence exists in the paper.** The decontamination procedure is described in Section 2.1 and Appendix C.5, but only at the question level (8-gram overlap). The paper does not analyze whether Gemini's reasoning traces contain benchmark-specific knowledge, nor does it evaluate s1-32B on held-out benchmarks that are provably not in any training corpus (e.g., newly constructed problems). The follow-up s1.1 (Appendix A) uses traces from DeepSeek R1, an open-weight model whose training data composition is partially known but not fully audited. Table 5 shows that s1.1 with R1 traces substantially outperforms s1 with Gemini traces (56.7% vs. 50.0% on AIME24 without BF), which could reflect either better distillation quality or greater contamination in R1's training data. The 59K-full model (Table 2) achieves 53.3% on AIME24 vs. s1K's 50.0%, with the 95% confidence interval [-13.3%, +20.0%] spanning zero — this is consistent with both "more data doesn't help" and "contamination is already saturated at 1K examples."

**Mitigation status.** The paper makes no attempt to test for conceptual leakage or to evaluate on contamination-free benchmarks. The 8-gram decontamination is standard practice but insufficient for reasoning benchmarks where problems may share mathematical structure without sharing exact wording. A stronger test would be to evaluate s1-32B on newly constructed benchmarks (e.g., AIME 2025, which is reported only for s1.1 in Table 5 and not for the main s1 model) or on problems from a held-out time period that post-dates the training of the source models. The s1.1 results on AIME 2025 (Table 5: 50.0% without BF, 50.0% with BF) are substantially lower than AIME 2024 (56.7% without BF, 56.7% with BF), which could indicate either that AIME 2025 is genuinely harder, that the new traces are less contaminated, or that the R1-distilled traces are less effective for forward transfer. The paper does not interpret this gap, and it remains an unexamined confound for all results.

## 7. Implications and Future Directions
- How this changes the field
  - Demonstrates that controlled, monotonic test‑time scaling does not require massive RL or complex search pipelines: a tiny, carefully selected SFT set plus a decoding trick suffices (Figures 1–2; Table 3).
  - Introduces a simple, standardized metric suite (`Control`, `Scaling`, `Performance`) that can become a common yardstick for compute‑vs‑accuracy studies (§3.2).
  - Reframes the “reasoning model” problem as partly an inference-time control problem, not only a training-data problem.
- Follow‑up research enabled
  - Better extrapolation: combine budget forcing with diversity in continuation cues (beyond “Wait”), or integrate temperature/frequency penalties to avoid loops (§6.2).
  - RL + budget forcing: test whether models trained with RL policies respond more productively to enforced longer thinking (§6.2).
  - Hybrid scaling: orchestrate sequential scaling within each trajectory and parallel tree search across trajectories (Figure 7).
  - Data curation science: automate the “quality–difficulty–diversity” triage and explore teacher mixtures; study how trace correctness/noise affects SFT efficacy (§5.1; §A).
  - Robust control: develop token‑ or step‑aware decoders that truly honor budgets without external forcing (Tables 12–13).
- Practical applications
  - Cost‑aware deployment: set per‑query compute budgets and only extend thinking when needed, improving latency/cost for easy questions while retaining high accuracy on hard ones (Figures 1, 4a).
  - Education and problem solving: competition math and STEM tutoring where reasoning transparency and adjustable depth are valued (Figure 5).
  - Systems with tight context limits: augment sequential scaling with lightweight parallel search or PRMs when longer contexts are required (Figure 7).

> Result highlights to remember:
> - Minimal SFT (1K examples) + budget forcing ⇒ consistent test‑time scaling and strong scores: 56.7 (AIME24), 93.0 (MATH500), 59.6 (GPQA) (Table 1; Figures 1–2).
> - Budget forcing achieves perfect control and best accuracy among tested methods (Table 3); “Wait” is the most effective continuation cue (Table 4).
> - Carefully curated “quality–difficulty–diversity” data selection is crucial; random or single‑axis selection drops AIME24 by 13–23 points (Table 2).
