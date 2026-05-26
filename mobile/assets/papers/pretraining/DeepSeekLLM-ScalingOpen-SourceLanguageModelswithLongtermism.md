# DeepSeek LLM: Scaling Open-Source Language Models with Longtermism

**ArXiv:** [2401.02954](https://arxiv.org/abs/2401.02954)

## 🎯 Pitch

DeepSeek LLM redefines how open-source language models are scaled by empirically deriving new, practical scaling laws for allocating compute between model size and data size, as well as for key hyperparameters like batch size and learning rate. By introducing a more accurate measure of model scale (non-embedding FLOPs per token) and systematically validating these guidelines at billion-scale across 2 trillion tokens, DeepSeek produces bilingual models that not only surpass LLaMA-2 70B on multiple benchmarks but also approach or exceed GPT-3.5’s performance in open-ended tasks. This work equips the open-source AI community with robust, reproducible strategies to maximize performance and efficiency for future large-scale model training.

---

## 1. Executive Summary

This paper introduces DeepSeek LLM, a series of open-source language models at 7B and 67B parameters pre-trained on 2 trillion tokens of primarily Chinese and English text, and further aligned through supervised fine-tuning and direct preference optimization. The core contribution is a systematic re-examination of **scaling laws for hyperparameters** (modeling the power-law relationship between compute budget and optimal batch size/learning rate) and **optimal model/data scaling-up allocation strategy** (using non-embedding FLOPs/token as a novel model scale representation that avoids approximation errors in prior work), finding that higher data quality shifts the allocation toward scaling the model more aggressively. DeepSeek LLM 67B surpasses LLaMA-2 70B across code, mathematics, and reasoning benchmarks, and the DeepSeek 67B Chat model outperforms GPT-3.5 in both Chinese and English open-ended evaluations, establishing that a carefully scaled bilingual open-source model can match or exceed closed-source performance when scaling laws are accurately calibrated.

## 2. Context and Motivation

### The Core Problem: Open-Source LLM Scaling Has Been Flying Blind

The fundamental problem this paper addresses is that **the open-source LLM community lacks a principled, empirically validated framework for how to scale up model training effectively.** While the past few years have seen an explosion of open-source models — LLaMA, Mistral, Qwen, Baichuan, and many others — the process by which practitioners decide how large to make their models, how much data to train on, and what hyperparameters to use has been largely ad-hoc, guided more by convention and hardware constraints than by rigorous scaling analysis.

This matters because scaling decisions have enormous practical consequences. If you allocate more compute to model size when data size should have been scaled instead — or vice versa — you waste millions of dollars of compute on suboptimal performance. The paper frames this explicitly in Section 1, noting that "the scaling laws described in previous literature presents varying conclusions, which casts a dark cloud over scaling LLMs." Without reliable scaling laws, every new model release is a gamble: will scaling from 7B to 67B parameters yield proportional improvements, or will the model hit diminishing returns that could have been avoided by training on more data instead?

The paper situates this problem within the specific context of the open-source ecosystem's maturation. Following LLaMA's release (Touvron et al., 2023a,b), the community standardized around fixed-size model configurations — 7B, 13B, 34B, and 70B parameters — and focused on producing the highest-quality model at each size class. But as the authors observe in the Introduction:

> "Following LLaMA, the open-source community has primarily focused on training fixed-size (7B, 13B, 34B, and 70B), high-quality models, often neglecting research exploration into LLM scaling laws."

This is a pointed critique: the community is optimizing within fixed bins without understanding whether those bins are themselves optimal allocations of compute. A 67B model trained on 2T tokens might perform well, but what if a 50B model trained on 3.2T tokens would perform better for the same total FLOPs? Without scaling laws, no one can answer this question with confidence.

### Why This Problem Is Important (Both Theoretically and Practically)

The importance operates on two levels:

**Theoretical significance.** Scaling laws are the closest thing the field has to a unified theory of language model performance. They describe how generalization error decreases predictably with increases in compute budget $C$, model scale $N$, and data scale $D$, where $C \approx 6ND$. If these relationships hold, they provide a scientific foundation for deciding how to invest compute. However, as Table 4 in the paper starkly illustrates, the two foundational works on scaling laws reached fundamentally different conclusions:

- Kaplan et al. (2020), analyzing models trained on OpenWebText2, found that model scaling exponent $a \approx 0.73$ and data scaling exponent $b \approx 0.27$ — meaning the compute budget should be allocated primarily to scaling model size.
- Hoffmann et al. (2022), analyzing models trained on MassiveText, found approximately the opposite: $a \approx 0.49$ and $b \approx 0.51$ — a roughly equal allocation between model and data.

These are not minor discrepancies. They imply qualitatively different strategies: should you train a bigger model on the same data, or train the same-sized model on more data? The fact that the two most influential works in scaling laws cannot agree on this fundamental question means that **practitioners have no reliable guide for their most expensive decisions.** The paper frames this explicitly: these conflicting conclusions raise "doubts about the general applicability of scaling laws." Without resolving this conflict, scaling LLMs remains more art than science.

**Practical impact (the "long-termist" thesis).** The paper's subtitle — "Scaling Open-Source Language Models with Longtermism" — signals a specific practical motivation. The authors are not just building a single model; they are building an infrastructure for continuous improvement. In a long-term project, you need to know not just what worked for *this* model, but what will work for the *next* model that is $2\times$, $5\times$, or $10\times$ larger. Specifically, the paper aims to answer:

- What hyperparameters (batch size, learning rate) should a model use at a target compute budget, without needing to run expensive grid searches at scale?
- Given a total FLOPs budget, how should it be split between model size and data scale?
- Can the performance of a 1000$\times$ larger model be accurately predicted from small-scale experiments?
- How does data quality change the answers to all of the above?

The practical stakes are enormous. A $10\times$ scaling mistake at the 67B scale represents months of GPU time and millions of dollars wasted. The paper's scaling law framework is designed to prevent exactly this kind of waste by providing testable, quantitative predictions *before* committing to large-scale training.

### Where Prior Approaches Fall Short

The paper identifies several specific gaps in prior scaling law research:

**1. Incomplete hyperparameter analysis.** Prior works on scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) "often lacked a complete description of hyperparameter settings, leaving it uncertain whether models under different compute budgets reached optimal performance" (Section 3, opening paragraph). This is a fundamental issue: if you don't know whether each model in your scaling curve was trained with optimal hyperparameters, you can't be sure that observed performance differences are due to model/data scaling choices versus suboptimal training recipes. The paper addresses this head-on by first establishing *hyperparameter scaling laws* — modeling how the optimal batch size and learning rate vary with compute budget — before fitting the model/data scaling curve.

The authors are explicit about the practical failure modes of prior recommendations:

> "Early works (Goyal et al., 2017; McCandlish et al., 2018; Shallue et al., 2019; Smith et al., 2017; Zhang et al., 2019) provided some empirical observations for setting batch size and learning rate, but we found these observations have limited applicability in our preliminary experiments."

This means that hyperparameter heuristics that work for one model family or training setup may not transfer to another — yet another reason why reliable, empirically fitted scaling laws specific to your training infrastructure are necessary.

**2. Flawed model scale representation.** This is one of the paper's most technically specific contributions, and it addresses a subtle but consequential measurement error in prior work. Previous scaling laws represented model scale using either:

- Non-embedding parameters $N_1$ (Kaplan et al., 2020): model parameters excluding the embedding and output layers
- Complete parameters $N_2$ (Hoffmann et al., 2022): all parameters including embeddings

The compute budget is then approximated as $C = 6ND$, but as the paper shows in Table 3 and the accompanying analysis in Appendix A.2, **both $6N_1$ and $6N_2$ can misrepresent the actual computational cost by up to 50%**, especially for small-scale models. The specific failure modes are explained in equations (2):

- $6N_1 = 72 n_{\text{layer}} d_{\text{model}}^2$ — this *omits* the attention operation's computational overhead entirely
- $6N_2 = 72 n_{\text{layer}} d_{\text{model}}^2 + 6 n_{\text{vocab}} d_{\text{model}}$ — this *includes* vocabulary computation, which contributes less to model capacity

The paper introduces non-embedding FLOPs/token $M$ as an alternative:

$$M = 72 n_{\text{layer}} d_{\text{model}}^2 + 12 n_{\text{layer}} d_{\text{model}} l_{\text{seq}}$$

This includes the attention overhead (the $12 n_{\text{layer}} d_{\text{model}} l_{\text{seq}}$ term accounts for attention FLOPs) but excludes vocabulary computation. The compute budget then becomes simply $C = MD$, which is exact rather than approximate. Table 3 quantifies the consequences: for a small model (8 layers, $d_{\text{model}} = 512$), $6N_1 / M = 0.43$ — a 57% underestimate — while $6N_2 / M = 1.32$, a 32% overestimate. These discrepancies narrow for larger models (95 layers, $d_{\text{model}} = 8192$ yields ratios of 0.92 and 0.94) but are still non-trivial, and they *systematically distort the fitted scaling curve*, especially at low compute budgets where small-scale experiments anchor the fit. Appendix A.2 (Figure 6) confirms this visually: using $6N_1$ as the model scale representation overestimates large-model performance, while using $6N_2$ underestimates it. Only $M$ achieves accurate predictions for the 7B and 67B models.

**3. Complete neglect of data quality's impact on scaling laws.** Prior scaling laws research treated data as a homogeneous quantity — more tokens = better, with no consideration of data quality. The paper introduces a novel finding that fundamentally complicates this picture:

> "the data quality significantly influences the optimal model/data scaling-up allocation strategy. The higher the data quality, the more the increased compute budget should be allocated to model scaling."

This is demonstrated empirically across three datasets: early in-house data, current in-house data, and OpenWebText2 (Table 4). The model scaling exponent $a$ varies from 0.450 (early, lower-quality data) to 0.578 (OpenWebText2, the highest-quality data). The intuition the authors offer is that "high-quality data usually implies logical clarity and less predictive difficulty after sufficient training. Therefore, it's more advantageous to scale up the model size when increasing compute budget." This means the optimal allocation strategy is **not universal** — it depends on your data mixture. A practitioner training on carefully curated, high-quality data should bias toward larger models, while someone training on noisier web data should allocate more budget to data quantity.

This finding also potentially *resolves* the Kaplan vs. Hoffmann discrepancy: they used different datasets (OpenWebText2 vs. MassiveText) with different quality profiles, which naturally leads to different optimal scaling exponents. The paper doesn't explicitly claim this reconciliation, but it's a clear implication.

### How This Paper Positions Itself

The paper positions itself not as proposing a fundamentally new architecture or training method, but as providing the **empirical foundation that enables principled open-source LLM scaling.** Key aspects of this positioning:

**1. As a scaling law calibration effort, not just a model release.** The paper is unusual among open-source LLM releases (LLaMA, Mistral, Qwen) in dedicating its most technically dense section (Section 3) to scaling laws rather than to model architecture or training details. The message is clear: this paper's primary contribution to the community is not just the model weights, but the *methodology* for determining how to train models at any scale. The authors frame this explicitly in the Introduction:

> "Our study aims to lay the groundwork for future scaling of open-source LLMs, paving the way for further advancements in this domain."

This is a "teach a person to fish" positioning — the scaling law findings are designed to be reusable by other teams building models at different scales or with different data mixtures.

**2. As a synthesis and correction of prior scaling law work.** The paper doesn't present its scaling laws as a completely new discovery, but as a **refinement** that accounts for factors prior work missed: hyperparameter optimization, accurate model scale representation, and data quality effects. Figure 4 and Figure 5 show the paper's fitted IsoFLOP curve and performance scaling curve, which explicitly build on the Chinchilla IsoFLOP methodology (Hoffmann et al., 2022) while correcting for the issues described above. The language is careful: "we revisit scaling laws in this section to address these uncertainties and ensure we are on the right path to efficiently scale-up compute."

**3. As a demonstration that open-source can match closed-source when scaling is done right.** The evaluation results in Section 5 serve a dual purpose: they validate the scaling law predictions (the 7B and 67B models perform as predicted, per Figure 5) *and* they demonstrate that a properly scaled open-source model can surpass LLaMA-2 70B and compete with GPT-3.5. The subtext is that closed-source models' apparent advantage may partly reflect better-understood scaling behavior rather than fundamentally superior technology. The paper doesn't make this argument explicitly, but the juxtaposition of careful scaling law analysis with competitive benchmark results implies it.

**4. As part of a long-term roadmap.** The Conclusion section (Section 6) explicitly frames DeepSeek LLM as a project, not a product:

> "DeepSeek LLM is a long-term project committed to advancing open-source language models. Soon, we will release our technique reports in code intelligence and Mixture-of-Experts (MoE), respectively. At present, we are constructing a larger and improved dataset for the upcoming version of DeepSeek LLM."

The scaling laws are presented as the foundation for this roadmap — the tool that will guide decisions about how to scale the next version, how to incorporate higher-quality data, and how to allocate compute between dense and sparse architectures. The "longtermism" in the title is not just a philosophical stance; it's a practical commitment to using scaling laws as the decision-making framework for iterative model improvement.

### The Specific Research Questions the Paper Sets Out to Answer

Implicit in the paper's structure are four concrete research questions that Section 3 tackles:

1. **Hyperparameter scaling laws:** How do the optimal batch size and learning rate change as compute budget increases? Can these relationships be modeled with simple power laws to enable extrapolation from small-scale experiments to large-scale training?

2. **Optimal model/data allocation:** Given an accurate model scale representation (non-embedding FLOPs/token $M$), what is the optimal split between model size and data quantity as compute budget increases? Can this predict the performance of models 1000$\times$ larger than the fitting experiments?

3. **Data quality interactions:** How does the quality of the pre-training data affect the optimal allocation strategy? Can differences in data quality explain the conflicting conclusions of Kaplan et al. (2020) and Hoffmann et al. (2022)?

4. **Practical deployment:** Given answers to (1)–(3), how well do the resulting models (7B and 67B) perform relative to existing open-source and closed-source baselines across a comprehensive evaluation suite?

## 3. Technical Approach

### 3.1 Reader Orientation

DeepSeek LLM is a family of open-source language models at 7B and 67B parameters, pre-trained on 2 trillion bilingual tokens and then aligned through fine-tuning. The core technical problem the paper solves is: **given a long-term project to build increasingly capable models, how do you determine the right model size, data quantity, and training hyperparameters at each scale without wasting enormous compute on trial and error?** The answer is a systematic measurement of scaling laws — empirical power-law relationships that let you predict optimal configurations for models 1000$\times$ larger than your small-scale experiments — combined with a novel finding that the quality of your training data fundamentally shifts the optimal allocation between model size and data quantity.

### 3.2 Big-Picture Architecture (Diagram in Words)

The paper has two distinct but connected technical systems:

1. **The Scaling Law Measurement Pipeline** — a set of controlled small-scale experiments (compute budgets from $10^{17}$ to $2\times10^{19}$ FLOPs) that measure how model performance varies with hyperparameters (batch size, learning rate), model size, and data quantity. This pipeline produces fitted power-law equations that predict optimal settings at any scale.

2. **The DeepSeek LLM Training and Alignment Pipeline** — the actual large-scale system that uses the scaling law predictions to configure and train the 7B and 67B models on 2 trillion tokens of bilingual data, then fine-tunes them via supervised fine-tuning (SFT) and direct preference optimization (DPO) to produce chat-capable models.

Information flows as follows: first, the scaling law pipeline runs ~10 model/data configurations at each of 8 different compute budgets, measuring validation loss → the results are fitted to power laws giving optimal hyperparameter formulae and model/data allocation ratios → these predictions are used to configure the 7B and 67B pre-training runs → the pre-trained base models are evaluated → SFT fine-tunes them on 1.5M instruction instances → DPO further aligns the chat models using preference data.

### 3.3 Roadmap for the Deep Dive

- **First**, hyperparameter scaling laws (Section 3.1 of the paper) — how optimal batch size and learning rate change with compute budget, and why this must be established before anything else. This is foundational because all subsequent scaling curve experiments depend on models being trained with near-optimal hyperparameters at each compute budget.
- **Second**, the model scale representation problem — why the paper replaces the standard model parameters metric with non-embedding FLOPs/token $M$, and exactly how $M$ is computed. This is needed before fitting the model/data scaling curve because using the wrong representation corrupts the fitted exponents.
- **Third**, the IsoFLOP-based optimal model/data scaling analysis — how the paper fits the scaling curve, what the fitted exponents mean operationally, and how the performance predictions extrapolate to 1000$\times$ larger compute budgets. This is the core result that guides the 7B and 67B model design.
- **Fourth**, the data quality finding — how the optimal allocation strategy shifts when using datasets of different quality, why this potentially explains the Kaplan vs. Hoffmann discrepancy, and what it means for practitioners.
- **Fifth**, the pre-training pipeline itself (data preparation, architecture choices, hyperparameters) — how the scaling law predictions are concretely instantiated in the actual 7B and 67B training runs.
- **Sixth**, the alignment pipeline (SFT and DPO) — how the base models are converted into chat models, including the specific data compositions, training configurations, and design choices (staged fine-tuning, repetition mitigation, system prompt effects).

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **scaling law calibration and model training paper** whose core technical insight is that scaling law exponents depend on data quality, and that using non-embedding FLOPs/token $M$ as the model scale representation eliminates systematic errors in prior work's optimal allocation estimates.

---

#### Hyperparameter Scaling Laws (How Batch Size and Learning Rate Scale with Compute)

The paper's first technical contribution is establishing that the **optimal batch size and learning rate follow predictable power-law relationships with compute budget**, and that these relationships can be fitted from small-scale experiments and extrapolated to large-scale training. This is a prerequisite for fitting the model/data scaling curve: if you don't know what hyperparameters to use at each compute budget, you can't be sure that observed performance differences are due to model/data allocation rather than suboptimal training recipes.

**The problem being solved.** When training models at different scales, practitioners face a hyperparameter selection problem: should the batch size increase with model size? Should the learning rate decrease? Early works (Goyal et al., 2017; McCandlish et al., 2018; Shallue et al., 2019; Smith et al., 2017; Zhang et al., 2019) offered empirical heuristics, but the paper states these "have limited applicability in our preliminary experiments" (Section 3), indicating that existing heuristics did not transfer to their training setup. Without reliable scaling rules, every new model scale requires expensive hyperparameter grid searches.

**Experimental design.** The authors conducted grid searches over batch size and learning rate on models trained with fixed compute budgets. They first ran a dense grid at a compute budget of $10^{17}$ FLOPs for a specific model size (177M FLOPs/token), as shown in Figure 2(a). The results demonstrate a crucial property:

> "the generalization error remains stable across a wide range of choices of batch sizes and learning rates. This indicates that near-optimal performance can be achieved within a relatively wide parameter space."

This is a practically important finding: it means hyperparameter selection does not need to be exact — there is a broad "plateau" of near-optimal settings, which makes the scaling laws more forgiving.

The authors then extended this analysis across multiple compute budgets ranging from $10^{17}$ to $2\times10^{19}$ FLOPs. A key methodological detail: they used the **multi-step learning rate scheduler** (described in Section 2.3) and reused the first training stage across experiments, which enabled efficient exploration of the hyperparameter space without retraining models from scratch for each configuration.

**Defining "near-optimal."** To account for the flatness of the loss landscape, the paper defines a tolerance threshold:

> "Considering the redundancy in the parameter space, we regarded the parameters used by models whose generalization error exceeded the minimum by no more than 0.25% as near-optimal hyperparameters."

So the "optimal" batch size for a given compute budget is not a single point estimate but a range — any batch size whose validation loss is within 0.25% of the minimum qualifies. The same logic applies to learning rate.

**Fitting the scaling relationships.** The paper fits separate power laws for optimal batch size $B$ and optimal learning rate $\eta$ as functions of compute budget $C$:

> $$\eta_{\text{opt}} = 0.3118 \cdot C^{-0.1250}$$
> $$B_{\text{opt}} = 0.2920 \cdot C^{0.3271}$$
>
> where $\eta_{\text{opt}}$ is the optimal learning rate, $B_{\text{opt}}$ is the optimal batch size, and $C$ is the compute budget in FLOPs.

**What these equations compute:** Given a target compute budget $C$ (how many total floating-point operations you plan to spend on training), the first equation outputs the learning rate that minimizes validation loss, and the second outputs the batch size that minimizes validation loss. The negative exponent on learning rate ($-0.1250$) means optimal learning rate *decreases* as compute budget grows — larger training runs need smaller learning rates. The positive exponent on batch size ($+0.3271$) means optimal batch size *increases* with compute budget — larger training runs can productively use larger batches.

**Why these specific forms:** Power laws are the standard functional form in scaling laws research because they capture the observed log-linear relationships between scale variables and performance. The specific exponents are purely empirical — they are fitted to the data, not derived from theory. The fitted constants (0.3118, 0.2920) are specific to the DeepSeek training infrastructure and optimizer settings (AdamW with $\beta_1=0.9$, $\beta_2=0.95$, weight_decay=0.1, multi-step learning rate scheduler), and the paper does not claim they are universal. However, the *qualitative* trends (learning rate decreases, batch size increases) are consistent with intuitive expectations when scaling up models.

**Validation at scale.** The authors validated these formulae on models with a $10^{20}$ compute budget — an order of magnitude beyond the fitting range. Figure 2(b) shows the results for a specific model size (2.94B FLOPs/token). The fitted parameters fall in the center of the near-optimal region, confirming that the power-law extrapolation remains accurate. Further validation comes from the DeepSeek 7B and 67B models themselves: the hyperparameters in Table 2 (batch sizes of 2304 and 4608, learning rates of $4.2\times10^{-4}$ and $3.2\times10^{-4}$) are consistent with the predicted values from the fitted formulae at those model scales.

**Limitations and open questions.** The paper acknowledges an important subtlety:

> "we have not yet considered the impact of factors beyond the compute budget $C$ on the optimal hyperparameters. This is inconsistent with some earlier works (Kaplan et al., 2020; McCandlish et al., 2018) which suggested that the optimal batch size can be modeled as being solely related to the generalization error $L$."

Furthermore, they observed that "in models with the same compute budget but different model/data allocations, the optimal parameter space varies slightly." This means the hyperparameter scaling laws presented are a first-order approximation — the true optimal batch size and learning rate also depend on how the compute budget is split between model size and data quantity, not just on the total budget. The paper leaves this interaction effect to future work.

---

#### The Model Scale Representation Problem ($M$ vs. $N_1$ vs. $N_2$)

Before fitting the model/data scaling curve, the paper addresses a measurement problem that it identifies as a significant source of error in prior scaling law work: **how to represent model "scale" in a way that accurately reflects computational cost.** This may seem like a bookkeeping detail, but the paper demonstrates that using the wrong representation introduces systematic errors of up to 50% in the estimated computational cost of small models, which distorts the fitted scaling curve and leads to incorrect predictions for large models.

**The problem with prior representations.** Previous works used two different model scale metrics:

- **Non-embedding parameters $N_1$** (Kaplan et al., 2020): the number of model parameters excluding the token embedding matrix and the output projection layer. The computational cost is approximated as $C \approx 6N_1D$.
- **Complete parameters $N_2$** (Hoffmann et al., 2022): all model parameters, including embeddings. The cost is approximated as $C \approx 6N_2D$.

Both approximations use the factor of 6 because each parameter participates in approximately 6 floating-point operations per token during training (2 FLOPs for the forward pass and 4 for the backward pass, assuming the standard factor of 2 for backward relative to forward).

**Why these fail.** The paper derives the exact relationship between parameter counts and actual FLOPs in Equation (2):

> $$6N_1 = 72 n_{\text{layer}} d_{\text{model}}^2$$
> $$6N_2 = 72 n_{\text{layer}} d_{\text{model}}^2 + 6 n_{\text{vocab}} d_{\text{model}}$$
> $$M = 72 n_{\text{layer}} d_{\text{model}}^2 + 12 n_{\text{layer}} d_{\text{model}} l_{\text{seq}}$$

where $n_{\text{layer}}$ is the number of transformer layers, $d_{\text{model}}$ is the model's hidden dimension, $n_{\text{vocab}}$ is the vocabulary size (102,400 for DeepSeek LLM), and $l_{\text{seq}}$ is the training sequence length.

The three representations differ in what they include and exclude:

- **$6N_1$** omits the attention operation's computational overhead entirely. The attention mechanism computes query-key dot products and value-weighted sums, which costs $12 n_{\text{layer}} d_{\text{model}} l_{\text{seq}}$ FLOPs per token but is not captured by parameter count.
- **$6N_2$** includes vocabulary computation ($6 n_{\text{vocab}} d_{\text{model}}$), which contributes less to model capacity per FLOP. The embedding lookup and output projection are relatively cheap operations that scale with vocabulary size rather than model depth.
- **$M$ (non-embedding FLOPs/token)** includes the attention overhead but excludes vocabulary computation. It directly measures the computational cost of the transformer operations that scale with model capacity.

**Quantifying the error.** Table 3 in the paper shows how severely the representations diverge across model scales. For a very small model (8 layers, $d_{\text{model}} = 512$, the kind used in scaling law experiments at low compute budgets):

- $6N_1 / M = 0.43$ — the Kaplan representation underestimates actual FLOPs by 57%
- $6N_2 / M = 1.32$ — the complete-parameter representation overestimates by 32%

These are not small errors. If you're fitting a scaling curve using data from small models and then extrapolating to large models, a systematic 30–57% mis-estimation of the compute budget at the lower end will bend the fitted curve in ways that corrupt the predicted exponents and the predicted performance at scale.

As model size increases (toward the 7B and 67B scales), the ratios improve but remain non-trivial. For a 95-layer model with $d_{\text{model}} = 8192$ (the approximate scale of DeepSeek 67B), $6N_1/M = 0.92$ and $6N_2/M = 0.94$. The attention overhead and vocabulary computation become proportionally smaller as the $72 n_{\text{layer}} d_{\text{model}}^2$ term (the feed-forward and projection FLOPs) dominates, but even a 6–8% systematic error affects the precise exponents needed for billion-dollar compute allocation decisions.

**Why using $M$ matters for the scaling curve fit.** The paper's key claim is that using $M$ leads to more accurate extrapolation. Appendix A.2 (Figure 6) provides the visual evidence:

- When $6N_1$ is used as the model scale representation: the fitted curve *overestimates* the validation loss of the DeepSeek 7B and 67B models (the blue stars in Figure 6a fall below the fitted line, meaning the models perform better than predicted). This is because $6N_1$ underestimates the compute spent on small models, making them appear more efficient than they are, which flattens the fitted curve and leads to pessimistic predictions for large models.
- When $6N_2$ is used: the fitted curve *underestimates* the validation loss (the blue stars fall above the fitted line — the model performs worse than predicted). $6N_2$ overestimates the compute spent on small models, making them appear less efficient, which steepens the fitted curve and leads to optimistic predictions.
- When $M$ is used: the predictions are accurate — the blue stars lie on the fitted curve in Figure 6c.

**The compute budget identity under $M$.** With $M$ representing non-embedding FLOPs/token and $D$ representing the number of training tokens, the total compute budget becomes exactly:

$$C = M \cdot D$$

where $C$ is in total FLOPs, $M$ is in FLOPs per token, and $D$ is the token count. This is simpler and more accurate than $C \approx 6ND$, eliminating the approximation error entirely.

**Operationally, what $M$ captures.** The formula $M = 72 n_{\text{layer}} d_{\text{model}}^2 + 12 n_{\text{layer}} d_{\text{model}} l_{\text{seq}}$ counts:

- The feed-forward network FLOPs (two matrix multiplies with dimensions $d_{\text{model}} \times \frac{8}{3}d_{\text{model}}$ and back, plus the SwiGLU gating, totaling $8 \cdot 2 \cdot d_{\text{model}}^2 = 16 d_{\text{model}}^2$ per layer, times the standard factor of 2 for backward, plus query/key/value/output projections in attention which are $4 \cdot 2 \cdot d_{\text{model}}^2 = 8 d_{\text{model}}^2$, giving approximately $72 n_{\text{layer}} d_{\text{model}}^2$ as the parameter-related FLOPs).
- The attention dot-product and weighted-sum operations: $12 n_{\text{layer}} d_{\text{model}} l_{\text{seq}}$ (the constant 12 arises from 2 operations — query-key multiplication and value-weighted sum — each costing $2 \cdot d_{\text{model}} \cdot l_{\text{seq}}$ in the forward pass, times 2 for backward, times 1.5 for the multi-head factor, simplified to $12 n_{\text{layer}} d_{\text{model}} l_{\text{seq}}$).

The paper's choice to exclude vocabulary computation from $M$ is deliberate: the embedding layer FLOPs ($6 n_{\text{vocab}} d_{\text{model}}$) scale with vocabulary size rather than model capacity, so including them would make $M$ less representative of the model's "thinking" capability per token.

---

#### IsoFLOP-Based Optimal Model/Data Scaling Analysis

With the hyperparameter scaling laws established and the $M$ representation in hand, the paper proceeds to the central scaling law question: **for a fixed total compute budget $C$, what split between model size $M$ and data quantity $D$ minimizes validation loss?** This is the inference-time analog of the Chinchilla optimal allocation problem (Hoffmann et al., 2022), but with the $M$ correction and with explicit attention to data quality effects.

**The formal optimization problem.** The paper defines the objective in Equation (3):

> $$M_{\text{opt}}(C), D_{\text{opt}}(C) = \underset{M, D \text{ s.t. } C = MD}{\arg\min} L(M, D)$$

where $M_{\text{opt}}(C)$ is the model scale (non-embedding FLOPs/token) that minimizes validation loss for a given compute budget $C$, $D_{\text{opt}}(C)$ is the corresponding optimal data scale (number of training tokens), and $L(M, D)$ is the validation loss (measured in bits-per-byte on a held-out set). The constraint $C = MD$ enforces that the total compute spent is fixed.

**What this equation computes:** For each possible total compute budget $C$, find the pair $(M, D)$ whose product equals $C$ that gives the lowest possible validation loss. If the optimal $M$ turns out to increase faster with $C$ than $D$ does, that means you should bias toward bigger models when given more compute. If they increase at the same rate, the allocation is balanced.

**Experimental methodology — the IsoFLOP approach.** To solve this optimization without training models at every possible $(M, D)$ pair, the paper follows the IsoFLOP profile methodology from Chinchilla (Hoffmann et al., 2022). The procedure:

1. **Select 8 different total compute budgets:** ranging from $10^{17}$ FLOPs (very small) to $3\times10^{20}$ FLOPs (approaching the scale of the 7B model).
2. **For each budget, design ~10 different allocations** of $M$ and $D$ that spend the same total FLOPs. For example, at a budget of $10^{18}$ FLOPs, you might try a small model trained on many tokens ($M$ small, $D$ large) and a larger model trained on fewer tokens ($M$ large, $D$ small), plus several intermediate options.
3. **Train each configuration with the optimal hyperparameters** predicted by Equation (1) — this is why the hyperparameter scaling laws are established first.
4. **Measure validation loss** on an independent validation set of 100M tokens, drawn from the same distribution as the training data.
5. **For each compute budget, identify the $(M, D)$ pair with minimum loss** — this is the empirical optimum for that budget.
6. **Fit power laws** to how $M_{\text{opt}}$ and $D_{\text{opt}}$ grow with $C$.

**The IsoFLOP curves.** Figure 4(a) shows the resulting IsoFLOP curves — one curve per compute budget, each showing how validation loss varies as you sweep the model size $M$ for fixed total compute. The U-shape of each curve confirms that there is an optimal allocation: too small a model underfits (loss is high because the model lacks capacity), and too large a model also underfits (loss is high because there aren't enough training tokens for the model to converge). The minimum of each U-shaped curve is the optimal $(M, D)$ for that budget.

**The fitted optimal scaling exponents.** From the minima across all 8 compute budgets, the paper fits power laws for how the optimal $M$ and $D$ scale with $C$. The results from Equation (4):

> $$M_{\text{opt}} = M_{\text{base}} \cdot C^a, \quad M_{\text{base}} = 0.1715, \quad a = 0.5243$$
> $$D_{\text{opt}} = D_{\text{base}} \cdot C^b, \quad D_{\text{base}} = 5.8316, \quad b = 0.4757$$

where $M_{\text{opt}}$ is the optimal non-embedding FLOPs/token for compute budget $C$, $D_{\text{opt}}$ is the optimal number of training tokens, $M_{\text{base}}$ and $D_{\text{base}}$ are fitted constants, and $a$ and $b$ are the scaling exponents.

**What these equations compute:** Given a total compute budget $C$ (say, $10^{22}$ FLOPs), the first equation tells you what model size you should build (in FLOPs/token), and the second tells you how many tokens you should train it on. For the current in-house dataset, the model scaling exponent $a = 0.5243$ and the data scaling exponent $b = 0.4757$ — meaning that when you increase your compute budget, roughly 52.4% of the additional compute should go to scaling the model, and 47.6% should go to scaling the data. This is a slightly model-biased allocation, but quite close to the balanced 50/50 split found by Hoffmann et al. (2022).

**Why the exponents sum to approximately 1.** The constraint $C = M \cdot D$ implies $D = C / M$. If $M \propto C^a$ and $D \propto C^b$, then $C = M \cdot D \propto C^{a} \cdot C^{b} = C^{a+b}$, which requires $a + b = 1$ for consistency. The paper's fitted exponents satisfy $0.5243 + 0.4757 = 1.0000$, confirming internal consistency.

**Extrapolating to the 7B and 67B models.** Figure 5 shows the performance scaling curve — the validation loss (bits-per-byte) as a function of total compute budget $C$, with each point representing a model trained at the optimal $(M, D)$ for its budget. The key result is that the DeepSeek 7B and 67B models (the blue stars) lie almost exactly on the fitted power-law line extrapolated from models with up to 1000$\times$ smaller compute budgets. The paper states:

> "Using small-scale experiments can accurately predict the performance of models with 1000× compute budget. This provides both confidence and guidance for training models on a larger scale."

**What "accurately predict" means concretely.** The validation loss (bits-per-byte) predicted for the 7B and 67B models from the scaling curve matches the actual measured validation loss within the line width of Figure 5. This is the practical payoff of using $M$ as the model scale representation: without this correction, the predictions would have been systematically biased (as shown in Appendix A.2, Figure 6).

---

#### The Data Quality Finding (Why Optimal Allocation Strategy Depends on Your Dataset)

Perhaps the paper's most conceptually significant scaling law contribution is the finding that **the optimal model/data allocation strategy is not universal — it depends on the quality of the pre-training data.** This finding emerges from fitting the scaling exponents on three different datasets and observing that they systematically vary with data quality.

**The three datasets compared** (Table 4):

1. **Early in-house data:** the first version of DeepSeek's pre-training corpus, before iterative quality improvements.
2. **Current in-house data:** the refined version used for the final DeepSeek LLM training runs, which the paper states has "higher data quality than early in-house data."
3. **OpenWebText2:** the dataset used in Kaplan et al. (2020)'s original scaling law study. The paper notes that "the quality of OpenWebText2 even surpasses the current in-house data, due to its smaller scale which allows for more meticulous processing."

**The fitted exponents for each dataset:**

| Dataset | Model exponent $a$ ($M_{\text{opt}} \propto C^a$) | Data exponent $b$ ($D_{\text{opt}} \propto C^b$) |
|---|---|---|
| Early in-house data | 0.450 | 0.550 |
| Current in-house data | 0.524 | 0.476 |
| OpenWebText2 | 0.578 | 0.422 |

**What the pattern means.** As data quality increases:
- The model scaling exponent $a$ increases (0.450 → 0.524 → 0.578)
- The data scaling exponent $b$ decreases (0.550 → 0.476 → 0.422)

Operationally, this means: **when your pre-training data is higher quality, you should allocate a larger fraction of additional compute to making the model bigger, and a smaller fraction to getting more data.** When your data is lower quality, you should bias toward getting more data (to overcome noise and redundancy) rather than scaling the model.

**The intuition.** The paper offers a plausible mechanism:

> "high-quality data usually implies logical clarity and less predictive difficulty after sufficient training. Therefore, it's more advantageous to scale up the model size when increasing compute budget."

In other words, high-quality data is information-dense — each token provides a clearer learning signal. A larger model can effectively absorb this signal and use the additional capacity to learn more nuanced patterns. Low-quality data contains more noise, redundancy, and inconsistency, so the marginal benefit of a larger model is lower — you're better off getting more data to average out the noise.

**Why this might resolve the Kaplan vs. Hoffmann discrepancy.** Table 4 places the DeepSeek results alongside the two foundational prior works:

- Kaplan et al. (2020) (using OpenWebText2): $a = 0.73$, $b = 0.27$ — very model-biased
- Hoffmann et al. (2022) (using MassiveText): $a = 0.49$, $b = 0.51$ — balanced
- DeepSeek on OpenWebText2: $a = 0.578$, $b = 0.422$ — intermediate

The DeepSeek results on OpenWebText2 are closer to Kaplan than to Hoffmann, but not identical — the improved methodology (hyperparameter optimization, $M$ representation) shifts the exponents. The key insight is that **data quality appears to be a major confounder** in scaling law studies. Different datasets naturally lead to different optimal allocation strategies, which means there is no single "universal" scaling law — you must fit the exponents on your specific data.

**Practical implication.** For anyone training an LLM: before committing to a model size and data quantity, you should run small-scale IsoFLOP experiments on *your actual pre-training data mixture* (or a representative sample). The exponents you fit will be specific to your data quality. If you've invested heavily in data curation and filtering, your exponents will likely be more model-biased (higher $a$) — meaning you can productively train larger models on the same quantity of data. If your data is noisier, you should scale data more aggressively.

---

#### The Pre-Training Pipeline (Data Preparation, Architecture, and Hyperparameters)

With the scaling laws providing the strategic blueprint, Section 2 of the paper describes how the 7B and 67B models are actually built. The pipeline has three stages: data preparation, architectural configuration, and training execution.

**Data preparation (Section 2.1).** The pre-training corpus is built from Common Crawl dumps and other sources, processed through a three-stage pipeline:

1. **Deduplication:** The paper adopts an "aggressive deduplication strategy, expanding the deduplication scope." Rather than deduplicating within individual Common Crawl dumps (which catches only intra-dump duplicates), they deduplicate across all 91 dumps in their collection. Table 1 shows the deduplication rates:
   - Single dump: 22.2% of documents removed as duplicates
   - 91 dumps: 89.8% removed
   
   The paper notes that "deduplicating across 91 dumps eliminates four times more documents than a single dump method." This cross-dump deduplication catches content that appears repeatedly across different crawls — a more thorough approach that increases the effective information density of the corpus.

2. **Filtering:** The paper describes this as "developing robust criteria for document quality assessment" involving "both linguistic and semantic evaluations, providing a view of data quality from individual and global perspectives." Specific filtering criteria are not detailed, but the implication is that both surface-level features (language detection, length, formatting) and content-level features (coherence, informativeness) are used.

3. **Remixing:** The final step "adjusts our approach to address data imbalances, focusing on increasing the presence of underrepresented domains." This ensures that niche but valuable content (e.g., technical documentation, academic papers) is adequately represented rather than being drowned out by more common web text.

The resulting corpus contains **2 trillion tokens, primarily in Chinese and English**, and is "continuously expanding" — the paper presents it as a living dataset that will grow in future versions.

**Tokenizer design.** DeepSeek LLM uses a Byte-level Byte-Pair Encoding (BBPE) tokenizer implemented with the HuggingFace `tokenizers` library. Key design decisions:

- **Pre-tokenization** prevents merging tokens across character category boundaries (newlines, punctuation, CJK symbols), following GPT-2's approach. This ensures that punctuation marks and CJK characters remain as atomic units rather than being merged with adjacent text.
- **Digits are split into individual numerals**, following LLaMA's practice. So "2024" becomes four separate tokens: "2", "0", "2", "4". This avoids vocabulary explosion from number combinations while maintaining the ability to represent arbitrary numbers.
- **Vocabulary size:** 100,000 conventional tokens, trained on a 24 GB multilingual corpus, plus 15 special tokens for a total of 100,015. For training efficiency, the model's embedding dimension is padded to 102,400 — a multiple of 128 that aligns with GPU memory alignment requirements.

**Architecture (Section 2.2).** DeepSeek LLM largely follows the LLaMA architecture but with specific modifications:

**Micro design (shared with LLaMA):**
- Pre-Norm structure with RMSNorm (Root Mean Square Layer Normalization): normalization is applied before each sub-layer (attention and FFN) rather than after, which improves training stability.
- SwiGLU activation function for the Feed-Forward Network (FFN): a gated variant of GLU that uses Swish as the activation, shown by Shazeer (2020) to outperform standard ReLU and GELU in transformers.
- Intermediate FFN dimension of $\frac{8}{3} d_{\text{model}}$: For every transformer layer, the FFN hidden dimension is $\frac{8}{3}$ times the model's hidden dimension. For 7B ($d_{\text{model}} = 4096$), the FFN hidden size is approximately 10,923.
- Rotary Position Embedding (RoPE): relative positional encoding that encodes position information through rotation of the query and key vectors, enabling better length generalization than absolute position embeddings.
- Grouped-Query Attention (GQA) for the 67B model only: instead of having $n_{\text{heads}}$ separate key and value heads, GQA uses $n_{\text{kv\_heads}} = 8$ key-value heads shared across groups of query heads. For the 67B model with 64 query heads and 8 KV heads, each KV head serves 8 query heads. This reduces the KV cache size during inference by a factor of 8, significantly improving memory efficiency. The 7B model uses standard Multi-Head Attention (MHA) with 32 heads and 32 KV heads.

**Macro design (DeepSeek-specific departures from LLaMA):**

- **Layer count:** DeepSeek 7B has 30 layers (LLaMA-2 7B has 32), and DeepSeek 67B has 95 layers (LLaMA-2 70B has 80). The paper states these adjustments "while maintaining parameter consistency with other open-source models, also facilitate model pipeline partitioning to optimize training and inference." Specifically, the 67B model expands parameters by going *deeper* rather than *wider* — the hidden dimension stays at 8192 (same as LLaMA-2 70B), but the layer count increases from 80 to 95. This is a deliberate choice:
  > "Unlike most works using Grouped-Query Attention (GQA), we expanded the 67B model's parameters in network depth rather than the common practice of widening the intermediate width of FFN layers, aiming for better performance."
  
  The rationale: deeper models have been shown to benefit more from GQA because the KV cache compression matters more when there are more layers accumulating cached states.

**Hyperparameter configuration (Section 2.3).** Table 2 summarizes the key training hyperparameters:

| Parameter | 7B | 67B |
|---|---|---|
| Layers ($n_{\text{layers}}$) | 30 | 95 |
| Hidden dimension ($d_{\text{model}}$) | 4096 | 8192 |
| Attention heads ($n_{\text{heads}}$) | 32 | 64 |
| KV heads ($n_{\text{kv\_heads}}$) | 32 | 8 |
| Context length | 4096 | 4096 |
| Sequence length | 4096 | 4096 |
| Batch size | 2304 | 4608 |
| Learning rate | $4.2 \times 10^{-4}$ | $3.2 \times 10^{-4}$ |
| Training tokens | 2T | 2T |

The batch sizes and learning rates are consistent with the scaling law predictions from Equation (1): the larger 67B model uses a larger batch size (4608 vs. 2304) and a smaller learning rate ($3.2 \times 10^{-4}$ vs. $4.2 \times 10^{-4}$), following the power-law trends.

**Training initialization and optimizer:**
- Standard deviation of weight initialization: 0.006
- Optimizer: AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight_decay = 0.1
- Gradient clipping: 1.0 (gradients are clipped so their L2 norm does not exceed 1.0)
- Precision: bf16 (Brain Floating Point 16-bit) for training, with fp32 gradient accumulation

**Multi-step learning rate scheduler.** This is a notable departure from the standard cosine scheduler used in most LLM training (including LLaMA). The multi-step scheduler has three phases:

1. **Warmup:** 2000 steps of linear warmup from 0 to the maximum learning rate.
2. **Phase 1 (first 80% of tokens):** Learning rate stays at the maximum value.
3. **Phase 2 (80% to 90% of tokens):** Learning rate decreases to 31.6% of the maximum.
4. **Phase 3 (90% to 100% of tokens):** Learning rate decreases to 10% of the maximum.

Figure 1(a) justifies this choice: comparing a multi-step scheduler vs. a cosine scheduler on a 1.6B parameter model trained on 100B tokens, the final training loss is essentially identical. However, the multi-step scheduler provides a critical practical benefit:

> "When adjusting the training scale while keeping the model size fixed, the multi-step learning rate scheduler allows for the reuse of training from the first phase, offering a unique convenience for continual training."

In other words, if you train a model on 80% of your target tokens and then decide you want to continue training (either on the same data or on new data), the multi-step scheduler lets you pick up exactly where you left off. With a cosine scheduler, the learning rate has already decayed significantly by that point, and continuing training at a low learning rate may not be effective. This design choice directly supports the paper's "longtermist" philosophy — it makes iterative, continual model improvement practical.

Figure 1(b) explores different proportions for the three phases (e.g., 70%/20%/10% vs. 80%/10%/10%) and finds "slightly better performance" with some alternatives, but the paper selects the 80%/10%/10% split to balance performance and reusability.

**Infrastructure (Section 2.4).** The training framework is HAI-LLM (High-flyer, 2023), an in-house system that integrates:

- **Parallelism strategies:** data parallelism (splitting batches across GPUs), tensor parallelism (splitting individual matrix multiplications across GPUs), sequence parallelism (splitting long sequences across GPUs), and 1F1B pipeline parallelism (interleaved forward and backward passes to minimize idle time in pipeline-parallel training).
- **Flash Attention:** the IO-aware attention algorithm that reduces memory bandwidth requirements, enabling longer sequences and larger batch sizes than standard attention implementations.
- **ZeRO-1 optimizer state sharding:** optimizer states (AdamW's momentum and variance buffers) are partitioned across data-parallel ranks, reducing per-GPU memory consumption. ZeRO-1 shards only optimizer states, not gradients or parameters (which ZeRO-2 and ZeRO-3 would shard).
- **Computation/communication overlap:** the backward pass of the last micro-batch is overlapped with the reduce-scatter operation in ZeRO-1, and GEMM (matrix multiply) computation is overlapped with all-gather/reduce-scatter in sequence parallelism. This minimizes the time GPUs spend waiting for inter-GPU communication.
- **Operator fusion:** LayerNorm, GEMM, and Adam update kernels are fused where possible, reducing kernel launch overhead.
- **In-place cross-entropy:** a custom CUDA kernel that converts bf16 logits to fp32 precision on-the-fly during the cross-entropy computation (rather than converting in GPU memory beforehand), computes the gradient, and overwrites the logits buffer with the gradient in-place. This avoids allocating additional memory for the fp32 conversion.
- **Checkpointing:** model weights and optimizer states are saved asynchronously every 5 minutes, limiting lost progress to at most 5 minutes in case of hardware failure. The paper explicitly notes that "these temporary model checkpoints are cleared up regularly to avoid consuming too much storage space." Resuming from a different 3D parallel configuration is supported, allowing training to continue on a different GPU allocation if the cluster load changes.
- **Evaluation:** vLLM (Kwon et al., 2023) is used for generative evaluation tasks, and continuous batching for non-generative tasks, avoiding manual batch size tuning and reducing token padding.

**Why these infrastructure choices matter for scaling laws.** The scaling law measurements require running many small-scale experiments across a range of model sizes and batch sizes. If the training infrastructure introduces overhead that varies with model scale (e.g., less efficient parallelism for small models), it could distort the fitted exponents. The paper's use of fused kernels, communication overlap, and ZeRO-1 sharding aims to make the computational cost scale predictably with $M$ and $D$, so that the measured validation loss truly reflects model capacity and data quantity rather than infrastructure artifacts.

---

#### The Alignment Pipeline (SFT and DPO)

Sections 4 and 5.1.2 describe the process of converting DeepSeek LLM base models into chat-capable models. The alignment pipeline has two stages: Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO).

**SFT data composition.** The instruction dataset comprises approximately 1.5 million instances in English and Chinese, split into:

- **Helpful data:** 1.2 million instances, with sub-distributions of:
  - 31.2% general language tasks (conversation, writing, summarization, question answering)
  - 46.6% mathematical problems
  - 22.2% coding exercises
- **Safety data:** 300K instances covering various sensitive topics (discrimination, illegal behavior, privacy, etc.)

The high proportion of math and code data (68.8% combined) reflects a deliberate strategy: the paper observes in the discussion (Section 5.5) that "the model's capabilities may be primarily focused on code completion and algebraic questions" after SFT, and acknowledges that "to develop a comprehensive understanding of mathematics and coding, it is crucial to incorporate a diverse range of data during the pre-training stage."

**SFT training configuration:**

- **7B model:** 4 epochs, learning rate $1\times10^{-5}$
- **67B model:** 2 epochs, learning rate $5\times10^{-6}$

The different epoch counts are explained by a key observation: "we observed the overfitting problem is serious on the 67B model." Larger models have more capacity to memorize the SFT data, so fewer epochs are needed before overfitting sets in. The paper also notes that "GSM8K and HumanEval are improved consistently for the 7B model, while the 67B model hits the upper bound soon" — the 67B model already approaches the ceiling of these benchmarks with fewer SFT steps.

**The repetition problem and staged fine-tuning.** The paper identifies a specific failure mode during SFT: the **repetition ratio** — the proportion of generated responses that "fail to terminate and instead endlessly repeat a sequence of text." The authors observe that "the repetition ratio tends to rise as the quantity of math SFT data increases" because "math SFT data occasionally includes similar patterns in reasoning" and "weaker models struggle to grasp such reasoning patterns, resulting in repetitive responses."

For the 7B model, the solution is **staged fine-tuning** (Table 12):

- **Stage 1:** Fine-tune on all data (general + math + code). This produces a repetition ratio of 2.0% (at temperature 0).
- **Stage 2:** Fine-tune only on conversational data (excluding math and code). This reduces the repetition ratio to 1.4% while maintaining benchmark performance (HumanEval stays at 48.2%, GSM8K drops only slightly from 63.9 to 63.0).

The second stage essentially "smoothes out" the model's generation behavior without erasing the math and code capabilities acquired in stage 1. The instruction-following capability also improves (IFEval score increases from 38.0 to 41.2), suggesting that the conversational fine-tuning helps the model adhere to output format expectations.

For the 67B model, the repetition ratio after stage 1 is already below 1%, so a second stage is deemed unnecessary (and would hurt benchmark scores).

**Why the staged approach works (the mechanism).** The paper offers an implicit explanation: the math and code SFT data teaches the model to follow chain-of-thought reasoning patterns, but these patterns can become "stuck" in loops when the model encounters a situation it doesn't fully understand. The conversational fine-tuning in stage 2 teaches the model when to *stop* generating — it learns to produce a complete response and terminate rather than continuing to generate reasoning steps indefinitely.

**Direct Preference Optimization (DPO) stage.** After SFT, the chat models undergo DPO training to further improve alignment:

**DPO data construction:**
- **Helpfulness data:** multilingual prompts covering creative writing, question answering, instruction following, etc. For each prompt, the DeepSeek Chat model generates multiple response candidates. The preferred and dispreferred responses are selected (the paper does not detail the selection criteria, but standard practice is to use human annotation or an LLM-as-judge to rank responses).
- **Harmlessness data:** similar construction process, focused on safety-related prompts.

**DPO training configuration:**
- 1 epoch, learning rate $5\times10^{-6}$, batch size 512
- Learning rate warmup and cosine scheduler

**What DPO achieves (and doesn't achieve).** The paper reports that "DPO can strengthen the model's open-ended generation skill, while engendering little difference in performance among standard benchmarks." Table 17 confirms this: DPO causes minimal changes across most standard benchmarks (MMLU: 71.1→70.9, GSM8K: 84.1→85.2, HumanEval: 73.8→71.3). The main gains are in open-ended quality, as measured by AlignBench (Table 7) and MT-Bench (Table 8), where the DPO model shows consistent improvements across categories.

**System prompt design.** DeepSeek Chat uses a system prompt adapted from LLaMA-2's, with the following text:
> "You are DeepSeek Chat, a helpful, respectful and honest AI assistant developed by DeepSeek. The knowledge cut-off date for your training data is up to May 2023. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

An intriguing finding emerges from Table 14: the system prompt improves MT-Bench performance for the 67B model (8.35→8.58) but slightly degrades performance for the 7B model (7.15→7.11). The paper explains: "larger models possess a better understanding of the intended meaning behind the system prompt, enabling them to follow instructions more effectively and generate superior responses. On the other hand, smaller models struggle to grasp the system prompt adequately, and the inconsistency between training and testing might negatively impact their performance." This is a practical insight: system prompts are not uniformly beneficial across model scales, and below a certain capability threshold, they may confuse rather than guide the model.

**Instruction data in pre-training — a negative result.** The paper reports an experiment where 5 million instruction data instances (primarily multiple-choice questions) were incorporated during the final 10% of pre-training. The result: "the base model did exhibit improved performance on the benchmark. However, the final outcomes were nearly identical to those achieved by adding the same data during the SFT stage." Furthermore, the paper argues that including multiple-choice data during pre-training leads to benchmark overfitting without improving genuine intelligence:

> "we have observed that this improvement does not extend to the model's performance on other evaluations that do not utilize the multiple-choice format... This suggests that users may not perceive the model as becoming more intelligent during conversational interactions, as these interactions involve generating responses rather than solving multiple-choice problems."

This is an important methodological caution: benchmark improvements from instruction data mixing may reflect format-specific learning rather than general capability gains. The paper's decision to exclude both instruction data from pre-training and multiple-choice data from fine-tuning is a deliberate choice to avoid "benchmark decoration" — inflating benchmark scores through data that doesn't translate to real-world performance.

## 4. Key Insights and Innovations

### Innovation 1: Data Quality as a Confounding Variable in Scaling Laws — a New Diagnostic Axis

The paper's most conceptually significant contribution is the finding that **the optimal model/data scaling-up allocation strategy is not universal — it depends on the quality of the pre-training data, and this dependence has been the hidden variable behind conflicting results in prior scaling law research.**

Before this work, the field treated scaling law exponents as fixed properties of neural language modeling. The two canonical studies — Kaplan et al. (2020) finding a ~0.73/0.27 model/data split and Hoffmann et al. (2022) finding ~0.49/0.51 — were seen as contradictory, requiring one to be "right" and the other "wrong." The implicit assumption was that scaling laws, like physical laws, should be invariant to the particular dataset used, as long as the dataset is "large enough" and representative.

DeepSeek LLM reframes this entirely. By fitting the scaling exponents on three different datasets — early in-house data, current in-house data, and OpenWebText2 — and observing that the model scaling exponent $a$ varies systematically from 0.450 to 0.578 (Table 4), the paper demonstrates that **data quality is a first-class variable in scaling laws, not a nuisance parameter.** The finding is not just that different datasets give different exponents (that would be trivial); it's that the exponents vary *monotonically* with data quality: as quality increases, the optimal strategy shifts toward allocating more additional compute to model size and less to data quantity.

This is a **diagnostic reframing, not just a new fact.** It transforms the Kaplan-vs-Hoffmann contradiction from a problem (whose numbers are wrong?) into evidence (they studied datasets of different quality, so their exponents *should* differ). The implication is that scaling law exponents are a **measurable property of a dataset**, not a constant of nature. This means future scaling law work cannot report exponents without characterizing the dataset used, and practitioners cannot adopt exponents from prior work without validating them on their own data.

The paper doesn't overclaim here — it offers only a tentative mechanism ("high-quality data usually implies logical clarity and less predictive difficulty after sufficient training") and explicitly states that "we will continue to pay close attention to the changes in data quality and its impact on scaling laws, and provide more analysis in future works." This is appropriate restraint for a finding that is clearly important but not yet fully explained. The significance is that it opens a new research axis: rather than asking "what are the scaling laws?", the field should now ask "how do scaling laws vary with data properties, and can we predict them from data characteristics without running expensive IsoFLOP experiments?"

The finding also has immediate practical bite. For organizations investing in data curation — cleaning, deduplication, quality filtering — the paper provides the first empirical evidence that these investments change *how they should allocate their scaling compute.* Better data doesn't just improve performance directly; it shifts the optimal architecture toward larger models. This is a non-obvious interaction that, prior to this paper, no one was accounting for in their scaling decisions.

---

### Innovation 2: Non-Embedding FLOPs/Token as a Model Scale Metric — Fixing a Silent Systematic Error

The paper's second key innovation is technically narrow — it's about how you *measure* model size for scaling law fitting — but its implications are broad: **prior scaling law studies have been using a flawed measurement instrument, and correcting this measurement changes the predicted optimal allocation and performance extrapolation.**

The field had settled on using parameter counts (either non-embedding $N_1$ or complete $N_2$) as the model scale variable, with the compute budget approximated as $C \approx 6ND$. This approximation is convenient — parameter count is easy to compute and compare across architectures — but the paper demonstrates that it introduces **systematic errors of up to 50%** in the estimated compute cost of small models (Table 3). Since scaling laws are fitted using small-scale experiments and extrapolated to large models, a systematic measurement error at the low end *bends the fitted curve* in ways that corrupt the predicted exponents.

What makes this more than a bookkeeping correction is that the paper shows the error is **directional and scale-dependent.** At small model sizes, $6N_1$ (Kaplan's metric) underestimates actual FLOPs by 57% because it omits attention computation, while $6N_2$ (Hoffmann's metric) overestimates by 32% because it includes vocabulary computation that contributes little to model capacity per FLOP. These errors shrink as models scale up (to ~6–8% at the 67B scale), but the *asymmetry* between small and large models distorts the fitted scaling curve. Appendix A.2, Figure 6 provides the visual evidence: using $6N_1$ makes the fitted curve overestimate large-model performance (predictions are pessimistic), while $6N_2$ makes it underestimate performance (predictions are optimistic). Only the proposed $M$ metric — non-embedding FLOPs/token, which includes attention overhead but excludes vocabulary computation — yields accurate predictions for the 7B and 67B models.

This is a **fundamental methodological refinement**, not an incremental improvement. It's analogous to discovering that a widely-used thermometer has a systematic calibration error that varies with temperature. Every conclusion drawn from that thermometer needs to be re-examined. The paper doesn't claim that prior scaling law work is invalidated — the qualitative conclusions (performance improves predictably with compute) still hold — but the *quantitative* exponents and optimal allocations are sensitive to this correction. The paper's fitted exponent of $a = 0.5243$ for its current dataset (a slightly model-biased allocation) reflects both its data quality and its corrected measurement methodology. Without the $M$ correction, the fitted exponents would have been different, and the performance predictions for the 7B and 67B models would have been biased.

The fact that this innovation is about *measurement* rather than *method* makes it easy to overlook, but it's arguably the paper's most rigorous contribution. It demonstrates that careful attention to operational definitions — what exactly does "model scale" mean, and how exactly is compute cost calculated? — can resolve apparent contradictions and produce more reliable predictions. This is the kind of contribution that improves the whole field's methodology, not just one team's models.

---

### Innovation 3: Hyperparameter Scaling Laws as a Prerequisite, Not an Afterthought

Prior scaling law work treated hyperparameters (batch size, learning rate) as something to *set* — often via heuristics or small-scale tuning — and then proceeded to fit the model/data scaling curve. The implicit assumption was that if you used "reasonable" hyperparameters at each scale, the resulting loss measurements would be close enough to optimal that the fitted scaling exponents would be reliable.

DeepSeek LLM inverts this. It establishes **hyperparameter scaling laws first** — modeling optimal batch size and learning rate as power-law functions of compute budget (Equation 1) — and treats this as a *prerequisite* for valid model/data scaling analysis, not a side detail. The logic is explicit: if you don't know that each model in your IsoFLOP curve was trained with near-optimal hyperparameters, you can't be sure whether observed performance differences are due to the model/data allocation or to suboptimal training recipes.

What makes this distinctive is the **empirical demonstration that optimal hyperparameters follow predictable power-law trends, but also that the "optimal" region is broad and forgiving.** Figure 2(a) shows that near-optimal performance can be achieved across a wide range of batch sizes and learning rates — a finding that is itself practically important because it means hyperparameter selection doesn't need to be exact. Figure 3 shows the fitted power-law trends: optimal batch size increases with compute budget ($B_{\text{opt}} \propto C^{0.3271}$) and optimal learning rate decreases ($\eta_{\text{opt}} \propto C^{-0.1250}$). These trends are intuitive — larger training runs can productively use larger batches and need smaller learning rates for stable convergence — but quantifying them as fitted power laws enables *extrapolation* from small-scale experiments to large-scale training without running expensive grid searches at each scale.

This is a **methodological innovation** rather than a theoretical one. It doesn't change how we think about scaling laws conceptually, but it changes how scaling laws should be *measured.* The paper is essentially arguing that prior work's failure to account for hyperparameter scaling may be another reason (alongside data quality and model scale representation) why Kaplan and Hoffmann reached different conclusions. Models at different compute budgets in those studies may not have been equally well-tuned, introducing noise or bias into the fitted exponents.

The paper is appropriately cautious about the limitations: it acknowledges that the fitted power laws may not capture interaction effects between hyperparameters and model/data allocation ("in models with the same compute budget but different model/data allocations, the optimal parameter space varies slightly"), and that the specific constants (0.3118, 0.2920) are specific to their training infrastructure. But the *qualitative* finding — that hyperparameter scaling laws exist, can be fitted from small-scale experiments, and should be established before fitting model/data scaling curves — is a transferable methodological contribution to the field.

---

### Innovation 4: Breaking the "Multiple-Choice Benchmark Trap" — Honest Evaluation Through Negative Results

This paper includes a meta-contribution that is rare in the LLM literature: a deliberate, publicly documented **decision to avoid practices that inflate benchmark scores without improving real-world performance, even when those practices are standard in the field.**

Specifically, the paper reports (in Section 5.5) that:
- Adding 20 million multiple-choice data instances during fine-tuning significantly boosted benchmark scores (MMLU: 49.4→60.9, C-Eval: 47.0→71.3, CMMLU: 49.7→73.8), but had **no effect** on generative evaluation tasks (TriviaQA, ChineseQA) that don't use the multiple-choice format.
- Similarly, incorporating instruction data during the final 10% of pre-training improved benchmark performance, but the gains were identical to simply adding that data during SFT — meaning the apparent improvement was not due to better pre-training, but to format-specific learning during fine-tuning.
- Based on these findings, the paper **explicitly excluded multiple-choice data from both pre-training and fine-tuning**, and excluded instruction data from pre-training.

This is an **innovation in research norms**, not in technical methods. The paper is making a statement about what constitutes valid evaluation: benchmark improvements that don't transfer to open-ended generation are "benchmark decoration" — they make numbers go up without making the model more useful. The authors frame this as a conscious choice:

> "We avoid benchmark decoration and dark secrets in all training stages."

The significance is that this paper provides **documented negative results that support this choice**, making it easier for other researchers to justify similar decisions. The field has long known anecdotally that training on multiple-choice data inflates multiple-choice benchmark scores, but having a clear empirical demonstration — with the specific magnitude of the effect quantified (MMLU +11.5 points, C-Eval +24.3 points) and the zero transfer to generative tasks confirmed — raises the burden of proof for anyone who *does* include such data. It shifts the default from "everyone does it, so it's fine" to "if you include multiple-choice data in training, you need to demonstrate that the gains reflect genuine capability improvements, not just format overfitting."

This is a **fundamental cultural contribution** rather than a technical one. It doesn't change how models are architected or trained, but it changes how their performance should be interpreted. In a field where benchmark leaderboards drive attention and funding, documenting and avoiding benchmark inflation is an act of intellectual honesty that raises the standard for the entire community. The paper doesn't frame it as an innovation, but in the current LLM landscape — where every percentage point on MMLU is competitively contested — it arguably is one.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on a comprehensive suite of public benchmarks spanning multiple domains, as enumerated in Section 5.1. These include: multi-subject multiple-choice datasets (MMLU, C-Eval, CMMLU); language understanding and reasoning datasets (HellaSwag, PIQA, ARC, OpenBookQA, BBH); closed-book QA (TriviaQA, NaturalQuestions); reading comprehension (RACE, DROP, C3); reference disambiguation (WinoGrande, CLUEWSC); language modeling (Pile); Chinese understanding and culture (CHID, CCPM); math (GSM8K, MATH, CMath); code (HumanEval, MBPP); and standardized exams (AGIEval). For open-ended evaluation, the paper uses AlignBench (Liu et al., 2023) for Chinese (683 questions across 8 primary categories) and MT-Bench (Zheng et al., 2023) for English (8 categories of multi-turn questions). Held-out evaluations include LeetCode Weekly Contest problems (126 problems from July–November 2023, with 20+ test cases each), the Hungarian National High-School Exam (33 problems, human-annotated), and the IFEval instruction-following benchmark (Zhou et al., 2023) with ~500 prompts covering 25 verifiable instruction types. Safety evaluation uses both an in-house test set of 2400 manually constructed questions across 5 safety categories (Table 10) and the Do-Not-Answer dataset (Wang et al., 2023) with 939 risk-categorized prompts.

- **Base model(s).** The primary models are DeepSeek LLM 7B (30 layers, $d_{\text{model}} = 4096$, 32 attention heads, trained on 2T tokens) and DeepSeek LLM 67B (95 layers, $d_{\text{model}} = 8192$, 64 attention heads with 8 KV heads via GQA, trained on 2T tokens), with detailed specifications in Table 2. For the alignment pipeline, the corresponding DeepSeek Chat models are produced via SFT and DPO. The primary baseline is LLaMA-2 at 7B and 70B (Touvron et al., 2023b), chosen because LLaMA has become "the de facto benchmark for architecture and performance among open-source models" (Section 1). For the chat model comparisons, additional baselines include GPT-3.5-turbo-0613, GPT-4-0613, GPT-4-1106-preview, and various open-source chat models including Qwen-14B-Chat, Baichuan2-13B-Chat, ChatGLM3-6B, InternLM-20B, and others. For code and math comparisons (Appendix A.4), baselines include Codex-001, StarCoder 16B, CodeGeeX2 6B, CodeLlama (7B/13B/34B), Wizard-Coder 34B, MetaMath 70B, WizardMath 70B, and ToRA-Code 34B.

- **Metrics.** The primary metric for most benchmarks is **accuracy** — the fraction of test instances where the model's selected or generated answer matches the ground truth. For Pile-test, the metric is **bits-per-byte (BPB)** — the average number of bits needed to encode each byte of the test corpus, with lower values indicating better language modeling. For DROP, the metric is **F1 score** — the harmonic mean of precision and recall over token-level answer matches. For perplexity-based evaluation (applied to HellaSwag, PIQA, WinoGrande, RACE, MMLU, ARC, OpenBookQA, CHID, C-Eval, CMMLU, C3, and CCPM), the model calculates the perplexity of each answer option given the prompt and selects the option yielding the lowest perplexity. For ARC and OpenBookQA specifically, "unconditional normalization" (Brown et al., 2020) is used, meaning the model scores each answer choice by how much it reduces perplexity relative to an empty context. For other perplexity-based datasets, length normalization is used. For generation-based evaluation (TriviaQA, NaturalQuestions, DROP, MATH, GSM8K, HumanEval, MBPP, BBH, AGIEval, CLUEWSC, CMath), the model generates free text using greedy decoding, and answers are parsed from the generated text using dataset-specific extraction logic. For open-ended evaluation, AlignBench uses GPT-4-0613 as a judge to rate response quality on a numerical scale, following the official rating templates provided with the benchmark. MT-Bench similarly uses GPT-4 as a judge to score multi-turn conversations. For LeetCode, the metric is pass@1 — the fraction of problems where the model's output passes all test cases on the first attempt. For the Hungarian Exam, human annotators grade model outputs following the official scoring rubric. For IFEval, the metric is prompt-level loose accuracy — the fraction of prompts where all verifiable instructions are satisfied. For safety evaluation on the in-house test set, each response is manually annotated as safe, unsafe, or refusal, and the safety rate is the proportion of responses rated as either safe or refusal. For Do-Not-Answer, the score is the fraction of prompts to which the model refuses to provide an unsafe response (higher is safer).

- **Baselines.** For base model evaluation (Table 5), the primary baseline is LLaMA-2 7B and 70B (Touvron et al., 2023b). For chat model evaluation (Table 6), the baseline is the corresponding base model evaluated in few-shot settings, with chat models evaluated in zero-shot settings where applicable. For Chinese open-ended evaluation (Table 7), baselines include GPT-4-1106-preview, GPT-4-0613, GPT-3.5-turbo-0613, and various open-source Chinese chat models (ChatGLM family, Erniebot, Qwen Chat, Baichuan Chat, InternLM Chat, Chinese-LLaMA-2 Chat, LLaMA-2-Chinese-Chat). For English open-ended evaluation (Table 8), baselines include GPT-4-1106-preview, GPT-3.5-turbo-0613, LLaMA-2-Chat (7B/13B/70B), Zephyr-Beta 7B, Xwin (70b/13b), and TÜLU 2+DPO 70B. For held-out evaluation (Table 9), baselines include GPT-4, ChatGLM3 6B, Baichuan2-Chat 13B, Yi-Chat 34B, and Qwen 72B Chat. For code comparison (Table 15), baselines include Codex-001, StarCoder 16B, CodeGeeX2 6B, CodeLlama (7B/13B/34B), and Wizard-Coder 34B. For math comparison (Table 16), baselines include MetaMath 70B, WizardMath 70B, and ToRA-Code 34B.

- **Generation budget / compute accounting.** The paper operates in two distinct evaluation regimes. For base model evaluation, all models (both DeepSeek and baselines) are evaluated using standard few-shot prompting protocols, with the number of in-context examples specified per benchmark (e.g., 5-shot for MMLU, 8-shot for GSM8K, 0-shot for HumanEval). Generation-based evaluation uses greedy decoding (temperature = 0), meaning there is no sampling budget to account for — each prompt produces exactly one deterministic output. For the scaling law experiments (Section 3), compute is measured in total training FLOPs, where $C = M \cdot D$ using the non-embedding FLOPs/token representation $M$ and the number of training tokens $D$. For hyperparameter scaling experiments, compute budgets range from $10^{17}$ to $2\times10^{19}$ FLOPs. For IsoFLOP experiments, 8 compute budgets ranging from $10^{17}$ to $3\times10^{20}$ FLOPs are used, with approximately 10 different model/data allocations tested at each budget. The 7B and 67B models each consume exactly $M \cdot 2\times10^{12}$ FLOPs for pre-training (since both train on 2T tokens), where $M$ is their respective non-embedding FLOPs/token. For SFT, the compute budget is not explicitly quantified, but the paper reports 4 epochs for the 7B model and 2 epochs for the 67B model on approximately 1.5M instruction instances. For DPO, the training runs for 1 epoch with a batch size of 512. For open-ended evaluation, generative tasks use temperature settings that vary by category: for AlignBench, role-playing, writing, and open-ended questions use temperature 0.7, while other tasks use temperature 0.1; for MT-Bench, standard generation parameters are used (not explicitly specified in the paper, but referenced as following the original benchmark protocol).

- **Cross-validation / statistical protocol.** The scaling law experiments involve fitting power laws to validation loss measured on an independent validation set of 100M tokens, "distributed similarly to the training set" (Section 3.2). The paper does not report cross-validation for the scaling law fits — the exponents are fitted once on the entire set of IsoFLOP experiments and then validated by comparing predicted performance for the 7B and 67B models against their actual measured validation loss (Figure 5). For benchmark evaluation, the paper uses standard train/test splits provided by each benchmark. For the in-house safety test set (2400 questions), a 20-person expert team from various disciplines constructed test cases and performed manual annotation, with cross-verification on annotation results. For the in-house ChineseQA test set mentioned in Appendix A.3 (Figure 7), no cross-validation details are provided. The paper does not report confidence intervals, standard errors, or statistical significance tests for any of the benchmark comparisons. The scaling law fits (Figures 3, 4, 5) are presented as point estimates without uncertainty quantification, though the fitted power laws themselves imply a specific functional form whose goodness-of-fit can be visually assessed from the figures.

### Main Quantitative Results

#### Base Model Evaluation Against LLaMA-2

The headline result from Table 5 is that **DeepSeek LLM 67B surpasses LLaMA-2 70B across a wide range of benchmarks, with particularly large margins in code, mathematics, and reasoning.** The most striking advantages include:

- **MATH (4-shot):** DeepSeek 67B achieves 18.7% vs. LLaMA-2 70B's 13.5% — a 5.2 percentage point absolute improvement, or 38.5% relative improvement.
- **GSM8K (8-shot):** DeepSeek 67B achieves 63.4% vs. LLaMA-2 70B's 58.4% — a 5.0 point gap.
- **HumanEval (0-shot):** DeepSeek 67B achieves 42.7% vs. LLaMA-2 70B's 28.7% — a 14.0 point gap, representing a 48.8% relative improvement.
- **MBPP (3-shot):** DeepSeek 67B achieves 57.4% vs. LLaMA-2 70B's 45.6% — an 11.8 point gap.
- **BBH (3-shot):** DeepSeek 67B achieves 68.7% vs. LLaMA-2 70B's 62.9% — a 5.8 point gap.
- **MMLU (5-shot):** DeepSeek 67B achieves 71.3% vs. LLaMA-2 70B's 69.0% — a narrower 2.3 point advantage.

On English language understanding tasks, the two models are more comparable. HellaSwag is tied at 84.0. DeepSeek leads on PIQA (83.6 vs. 82.0), ARC-Easy (76.9 vs. 76.5), and NaturalQuestions (36.6 vs. 36.1), while LLaMA-2 leads on WinoGrande (80.4 vs. 79.8), RACE-Middle (70.1 vs. 69.9), RACE-High (54.3 vs. 50.7), and DROP (69.2 vs. 67.9). The paper notes that "DeepSeek models are pre-trained on 2T bilingual corpus, [yet] they show comparable performance on English language understanding benchmarks with LLaMA2 models, which also consume 2T tokens but focus on English" — the implication being that the bilingual training does not significantly degrade English performance.

On Chinese benchmarks, the advantage is dramatic, as expected given LLaMA-2's English-focused training:

- **CHID (0-shot):** DeepSeek 67B achieves 92.1% vs. LLaMA-2 70B's 55.5% — a 36.6 point gap, reflecting the idiom-intensive nature of this task which requires substantial Chinese pre-training exposure.
- **C-Eval (5-shot):** DeepSeek 67B achieves 66.1% vs. LLaMA-2 70B's 51.4% — a 14.7 point gap.
- **CMMLU (5-shot):** DeepSeek 67B achieves 70.8% vs. LLaMA-2 70B's 53.1% — a 17.7 point gap.

An interesting cross-lingual transfer phenomenon is noted: LLaMA-2 70B achieves 53.9% on CMath (Chinese math) despite no Chinese pre-training, suggesting that "certain fundamental abilities, such as mathematical reasoning, can be effectively transferred across languages." DeepSeek 67B achieves 63.0% on the same benchmark.

At the 7B scale, the pattern is similar but with smaller absolute gaps. DeepSeek 7B leads LLaMA-2 7B on GSM8K (17.4% vs. 15.5%), HumanEval (26.2% vs. 14.6%), MBPP (39.0% vs. 21.8%), and MATH (6.0% vs. 2.5%). On Chinese tasks, the gap is larger: C-Eval (45.0% vs. 33.9%), CMMLU (47.2% vs. 32.6%), and CHID (89.3% vs. 37.9%). The paper observes that "the advantage of DeepSeek 67B over LLaMA2 70B is larger than that of DeepSeek 7B over LLaMA2 7B," attributing this to "the greater influence of language conflict on smaller models" — meaning that smaller models with less capacity struggle more to maintain performance in both languages simultaneously.

The Pile-test BPB metric shows DeepSeek models achieving slightly better language modeling performance: DeepSeek 7B at 0.725 vs. LLaMA-2 7B at 0.741, and DeepSeek 67B at 0.642 vs. LLaMA-2 70B at 0.649. These are small gaps but consistent in DeepSeek's favor.

Appendix A.3 (Figure 7) shows benchmark metric curves across training steps, revealing that performance on most benchmarks "improves consistently from the start to the end of training" on the 2T tokens. The paper notes that "the performance will further be improved if the training continues," suggesting the 2T token budget may not be saturating the models' capacity.

#### Chat Model Evaluation (Base vs. Chat)

Table 6 compares DeepSeek Chat models against their base counterparts. The key finding is that **SFT significantly improves performance on math and code tasks, while causing performance drops on certain cloze/sentence-completion tasks**, and the overall pattern is consistent across model scales.

**Substantial improvements after SFT:**

- **GSM8K:** 7B Chat (0-shot) achieves 63.0% vs. 7B Base (8-shot) at 17.4%; 67B Chat (0-shot) achieves 84.1% vs. 67B Base (8-shot) at 63.4%. These are improvements of 45.6 and 20.7 points respectively, though the shot count difference (0-shot for chat vs. 8-shot for base) should be noted — the chat models are evaluated in a setting more aligned with their intended use.
- **MATH:** 7B Chat (0-shot) achieves 15.8% vs. 7B Base (4-shot) at 6.0%; 67B Chat (0-shot) achieves 32.6% vs. 67B Base (4-shot) at 18.7%.
- **HumanEval:** 7B Chat achieves 48.2% vs. 7B Base at 26.2%; 67B Chat achieves 73.8% vs. 67B Base at 42.7%.
- **BBH:** 7B Chat achieves 42.3% vs. Base at 39.5%; 67B Chat achieves 71.7% vs. Base at 68.7%.

The paper's explanation is that "the base model was initially underfitted for these tasks, and the SFT stage has learned additional knowledge in coding and mathematics through the extensive SFT data" — the SFT dataset being 68.8% math and code combined.

**Performance drops after SFT:**

- **HellaSwag:** 7B Chat drops to 68.5% from 75.4%; 67B Chat drops to 75.7% from 84.0%. The paper attributes this to cloze/sentence-completion tasks being better suited to "pure language models."
- **CHID:** 7B Chat drops to 64.9% from 89.3%; 67B Chat drops to 72.6% from 92.1%.
- **WinoGrande:** 7B Chat drops to 66.9% from 70.5%; 67B Chat drops to 76.0% from 79.8%.

**Knowledge-related tasks show mixed patterns.** TriviaQA improves for 67B Chat (81.5% vs. 78.9%) but slightly declines for 7B Chat (57.9% vs. 59.7%). MMLU is essentially flat in the zero-shot chat setting compared to the 5-shot base setting (7B: 49.4% vs. 48.2%; 67B: 71.1% vs. 71.3%). The paper argues that "minor fluctuations" in knowledge tasks "do not indicate the acquisition or loss of knowledge after SFT" — rather, "the value of SFT lies in the ability to learn to achieve comparable scores to the base model's few-shot setting in the chat model's zero-shot setting, which is aligned with real scenarios."

#### Open-Ended Evaluation Results

**Chinese open-ended evaluation (AlignBench, Table 7).** The **DeepSeek 67B Chat DPO model scores 6.69 overall, placing it third behind only GPT-4-1106-preview (8.01) and GPT-4-0613 (7.53).** It significantly outperforms GPT-3.5-turbo-0613 (6.08) by 0.61 points and the next-best Chinese open-source model (chatglm-turbo at 6.24) by 0.45 points.

Breaking down by category:
- **Reasoning (推理):** DeepSeek 67B Chat DPO scores 5.77, behind GPT-4-1106-preview (7.73) and GPT-4-0613 (7.47) but ahead of GPT-3.5-turbo-0613 (5.35) and all other Chinese models. The math sub-score is 6.13, competitive with GPT-3.5 (5.68) and substantially ahead of the next Chinese model (chatglm-turbo at 4.74). The logic sub-score is 5.41, slightly ahead of GPT-3.5 (5.02).
- **Language (语言):** DeepSeek 67B Chat DPO scores 7.60, which approaches GPT-4-0613 (7.59) and exceeds GPT-4-1106-preview (8.29) in specific subcategories. Notably, in the "Fundamental Tasks" (基本任务) category, DeepSeek's score of 7.29 outperforms GPT-4-0613 (7.81 — wait, this is inconsistent; let me recheck. Table 7 shows DeepSeek DPO scoring 7.29 in "Fund." vs. GPT-4-0613's 7.81, so DeepSeek does not outperform here. The paper states: "For the basic Chinese Language tasks, our model is in the first tier among all models, and the Chinese fundamental language ability of our DPO model is even higher than the newest version of GPT-4" — but the table shows DeepSeek DPO at 7.29 in "Fund." vs. GPT-4-1106-preview at 7.99. This appears to be a misreading by the paper; the table shows GPT-4-1106-preview at 7.99 and GPT-4-0613 at 7.81, both above DeepSeek DPO's 7.29. The "first tier" claim might refer to being competitive rather than superior.) In Chinese Understanding (中文理解), DeepSeek DPO scores 7.47, exceeding GPT-4-0613 (6.93) and approaching GPT-4-1106-preview (7.33). In Open-Ended Questions (综合问答), DeepSeek DPO scores 7.82, ahead of GPT-3.5 (7.29) but behind GPT-4-0613 (7.42) and GPT-4-1106-preview (8.61). In Text Writing (文本写作), DeepSeek DPO scores 7.51, behind GPT-4 family (7.93–8.67). In Role-Playing (角色扮演), DeepSeek DPO scores 7.83, ahead of GPT-3.5 (7.28) and GPT-4-0613 (7.51), and close to GPT-4-1106-preview (8.47). In Professional Ability (专业能力), DeepSeek DPO scores 7.71, behind GPT-4-0613 (7.94) and GPT-4-1106-preview (8.65), but ahead of GPT-3.5 (6.77).

The DPO model consistently outperforms the SFT-only model across almost all subcategories: Overall improves from 6.43 to 6.69, Reasoning from 5.75 to 5.77, Language from 7.11 to 7.60. The improvement in Language tasks is particularly notable (7.11 → 7.60), with gains in Chinese Understanding (6.52 → 7.47) and Open-Ended Questions (7.20 → 7.82).

**English open-ended evaluation (MT-Bench, Table 8).** The **DeepSeek LLM 67B Chat DPO model achieves an average score of 8.76, placing second only behind GPT-4-1106-preview (9.26).** It surpasses GPT-3.5-turbo-0613 (8.39), LLaMA-2-Chat 70B (6.86), and all other open-source baselines. The SFT-only chat model scores 8.35, essentially tying GPT-3.5-turbo (8.39) and substantially ahead of the next best open-source model (TÜLU 2+DPO 70B at 7.89).

By category for the DPO model:
- **Reasoning:** 9.05, exceeding GPT-3.5 (6.20) by a wide margin and approaching GPT-4-1106-preview (8.10). This is the largest relative improvement from SFT (8.00 → 9.05).
- **Coding:** 6.75, ahead of GPT-3.5 (7.05 — actually behind; DeepSeek DPO at 6.75 vs. GPT-3.5 at 7.05, so GPT-3.5 leads here) and substantially ahead of LLaMA-2-Chat 70B (3.15). The SFT-only model scores 7.35, meaning DPO actually reduces coding performance (7.35 → 6.75).
- **Math:** 6.65, slightly behind GPT-3.5 (7.05) but well ahead of LLaMA-2-Chat 70B (3.30). SFT-only is 6.25, so DPO improves math slightly.
- **Extraction:** 9.30, behind GPT-4-1106-preview (9.90) but ahead of GPT-3.5 (9.00).
- **Humanities:** 9.80, close to GPT-4-1106-preview (9.95) and GPT-3.5 (9.95).
- **STEM:** 9.70, ahead of GPT-3.5 (9.55) and close to GPT-4-1106-preview (9.90).
- **Roleplay:** 9.10, ahead of GPT-3.5 (8.65) but behind GPT-4-1106-preview (9.50).
- **Writing:** 9.75, tied with GPT-3.5 (9.65) and approaching GPT-4-1106-preview (9.70).

The SFT-only model (8.35 average) shows a different profile: Coding (7.35) is notably strong, but Reasoning (8.00) and Writing (9.30) are lower than the DPO version. DPO seems to specifically boost Reasoning and Humanities while leaving Coding essentially unchanged or slightly degraded.

#### Held-Out Evaluation Results

Table 9 presents results on benchmarks designed to be immune to training data contamination:

**LeetCode Weekly Contest (126 problems, pass@1):**
- GPT-4: 48.4%
- DeepSeek LLM 67B Chat: 17.5%
- Qwen 72B Chat: 12.7%
- Yi-Chat 34B: 7.9%
- DeepSeek LLM 7B Chat: 4.7%
- ChatGLM3 6B: 2.4%
- Baichuan2-Chat 13B: 1.6%

DeepSeek 67B leads open-source models by a substantial margin (17.5% vs. 12.7% for Qwen 72B), but all models are far behind GPT-4. The paper notes that the performance gap between large and small models is "significant" on these out-of-domain coding problems, even though "certain small models achieve promising results on conventional benchmarks." The example given: ChatGLM3 scores 52.4 on MBPP (a standard code benchmark, comparable to DeepSeek 67B's 57.4 from Table 5) but only 2.4 on LeetCode — a 50-point drop that reveals benchmark overfitting.

**Hungarian National High-School Exam (33 problems, human-scored):**
- GPT-4: 68
- DeepSeek LLM 67B Chat: 58
- Qwen 72B Chat: 52
- Yi-Chat 34B: 39
- ChatGLM3 6B: 32
- DeepSeek LLM 7B Chat: 28.5
- Baichuan2-Chat 13B: 19.5

DeepSeek 67B leads open-source models (58 vs. Qwen 72B's 52) and is 10 points behind GPT-4. The gap between 67B and 7B (58 vs. 28.5) is larger than on standard math benchmarks (GSM8K: 84.1 vs. 63.0, MATH: 32.6 vs. 15.8), suggesting that the held-out nature of the Hungarian exam better discriminates model capability.

**IFEval (instruction following, prompt-level loose accuracy):**
- GPT-4: 79.3%
- DeepSeek LLM 67B Chat: 55.5%
- Qwen 72B Chat: 50.8%
- Yi-Chat 34B: 48.4%
- Baichuan2-Chat 13B: 44.5%
- DeepSeek LLM 7B Chat: 41.2%
- ChatGLM3 6B: 29.7%

DeepSeek 67B leads open-source (55.5 vs. Qwen's 50.8) but trails GPT-4 by 23.8 points. The 7B chat model (41.2%) outperforms ChatGLM3 (29.7%) and is competitive with Baichuan2-Chat 13B (44.5%), which the paper notes is "relatively commendable" given that DeepSeek 7B "falls behind other smaller language models on standard benchmarks."

The paper draws a broader conclusion from these held-out results: "total computing plays a crucial role" — larger models consistently outperform smaller ones on tasks that weren't seen during training, even when smaller models have competitive standard benchmark scores. The 7B and 67B comparison is illustrative: "DeepSeek 7B and 67B models utilize the same training pipeline, but there is a significant disparity in their performance. Through our subjective evaluation, we have observed a notable discrepancy in intelligence across various tasks when scaling model size to 67B."

#### Safety Evaluation Results

**In-house safety test set (Table 10):** The DeepSeek 67B Chat model is evaluated on 2400 manually constructed safety questions across 5 categories. The results, expressed as the number of safe answers (including model refusals) out of total cases:

- Discrimination and Prejudice Questions: 486/500 (97.2% safe)
- Infringement of Others' Legal Rights: 473/500 (94.6% safe)
- Trade Secrets and Intellectual Property: 281/300 (93.7% safe)
- Illegal and Non-compliant Behavior: 290/300 (96.7% safe)
- Other Safety Issues: 767/800 (95.9% safe)

Overall, the model provides safe responses on approximately 95.8% of test cases (2297/2400 by summing the numerators). The paper notes that the expert team constructed diverse safety issues "through means such as inducement, role-playing, multi-turn dialogues, preset positions, and etc.," and that annotators performed three-category annotation (safe, unsafe, refusal) with cross-verification.

**Do-Not-Answer benchmark (Table 11):** The DeepSeek 67B Chat model scores 97.8, which the paper claims is "higher than both ChatGPT and GPT-4." The full rankings:

- LLaMA-2-7B-Chat: 99.4
- Claude: 98.3
- DeepSeek 67B Chat: 97.8
- ChatGPT: 97.7
- GPT-4: 96.5
- Vicuna-7B: 94.9
- ChatGLM2: 92.9

The paper states this "places it amongst the ranks of the safest models." However, the very high score of LLaMA-2-7B-Chat (99.4, the highest in the table) and the fact that DeepSeek is slightly behind Claude (98.3) suggest that this metric may be dominated by refusal behavior — models that refuse more questions get higher scores, which may not perfectly correlate with nuanced safety judgment. DeepSeek's score is only 0.1 points above ChatGPT's 97.7, a negligible difference.

#### Code and Math-Specific Comparisons (Appendix A.4)

**Code comparison (Table 15):** Despite being a general-purpose model, DeepSeek LLM Base 67B achieves competitive performance with code-specific models:
- HumanEval: 42.7% vs. CodeLlama 34B at 48.2% and StarCoder 16B at 36.0%
- MBPP: 37.2% vs. CodeLlama 34B at 41.0%
- The "Python" and "Multilingual" columns show 57.4% for DeepSeek 67B Chat on Python coding tasks, ahead of CodeLlama 34B (55.2%)

The DeepSeek 67B Chat model (after SFT) reaches 73.8% on HumanEval, matching Wizard-Coder 34B (73.2%) and substantially exceeding CodeLlama 34B (48.2%).

**Math comparison (Table 16):** Using chain-of-thought reasoning:
- GSM8K: DeepSeek 67B Chat CoT at 84.1% vs. MetaMath 70B at 82.3% and WizardMath 70B at 81.6%
- MATH: DeepSeek 67B Chat CoT at 32.6% vs. MetaMath at 26.6% and WizardMath at 22.7%
- MGSM-zh (Chinese math): 74.0% vs. MetaMath at 66.4%
- CMath: 80.3% vs. MetaMath at 70.9%

Using tool-integrated reasoning (program-aided math solving):
- GSM8K: 86.7% vs. ToRA-Code 34B at 80.7%
- MATH: 51.1% vs. ToRA-Code at 50.8%
- MGSM-zh: 76.4% vs. ToRA-Code at 41.2%
- CMath: 85.4% vs. ToRA-Code at 53.4%

The tool-integrated approach yields significant gains on Chinese math benchmarks (MGSM-zh and CMath) where the gap with ToRA is 35+ points.

#### Benchmark Evolution During Training (Appendix A.3, Figure 7)

Figure 7 shows benchmark metrics as a function of training steps for DeepSeek LLM Base. The key observation is that performance improves "consistently from the start to the end of training" across all benchmarks shown, with no visible plateau or saturation at the 2T token mark. The paper explicitly states: "We believe the performance will further be improved if the training continues." This is significant because it validates one aspect of the scaling law predictions — the models are not yet at the "compute-optimal" stopping point predicted by the Chinchilla-style analysis, meaning there's headroom for further gains by training on more data.

#### DPO Stage Impact on Benchmarks (Appendix A.5, Table 17)

Table 17 compares DeepSeek 67B Chat before and after DPO. The key finding is that **DPO causes negligible changes across standard benchmarks**, consistent with the paper's characterization that DPO "engenders little difference in performance among standard benchmarks":

- MMLU: 71.1 → 70.9 (−0.2)
- GSM8K: 84.1 → 85.2 (+1.1)
- MATH: 32.6 → 30.2 (−2.4)
- HumanEval: 73.8 → 71.3 (−2.5)
- BBH: 71.7 → 70.8 (−0.9)
- AGIEval: 46.4 → 46.1 (−0.3)
- C-Eval: 65.2 → 64.3 (−0.9)
- CMMLU: 67.8 → 68.2 (+0.4)
- HellaSwag: 75.7 → 76.1 (+0.4)
- TriviaQA: 81.5 → 82.9 (+1.4)
- NaturalQuestions: 47.0 → 48.8 (+1.8)

The largest absolute change is −2.5 on HumanEval and −2.4 on MATH — small enough to be within the range of run-to-run variance. This confirms that DPO's benefits are primarily in open-ended generation quality (as measured by AlignBench and MT-Bench) rather than in knowledge or reasoning benchmarks. The slight degradation on some reasoning tasks (MATH, HumanEval, BBH) is consistent with the known phenomenon that alignment tuning can reduce benchmark performance in exchange for improved instruction-following and safety.

### Ablation Studies and Robustness Checks

**Two-Stage SFT vs. Single-Stage SFT (Section 5.5, Table 12):** For the 7B model, stage 1 (all data) produces a repetition ratio of 2.0% and IFEval of 38.0. Stage 2 (conversational data only, excluding math and code) reduces the repetition ratio to 1.4% (a 30% relative reduction) and improves IFEval to 41.2 (+3.2 points), while maintaining HumanEval at 48.2 and GSM8K at 63.0 (down only 0.9 from 63.9). This demonstrates that the staged approach successfully decouples math/code capability acquisition from conversation quality: the model retains the reasoning patterns learned in stage 1 while learning to terminate responses appropriately in stage 2. For the 67B model, a second stage "hurts the model score on the benchmark" and the repetition ratio is already below 1% after stage 1, so only single-stage SFT is used. The scale-dependent effectiveness is notable: the smaller model benefits from the staged approach, while the larger model does not need it — larger capacity appears to provide inherent protection against repetition.

**Multiple-Choice Data in Fine-Tuning (Section 5.5, Table 13):** The paper reports on an experiment adding 20 million Chinese multiple-choice data instances during SFT. The impact on multiple-choice benchmarks is dramatic: MMLU improves from 49.4 to 60.9 (+11.5), C-Eval from 47.0 to 71.3 (+24.3), and CMMLU from 49.7 to 73.8 (+24.1). However, this improvement is format-specific: TriviaQA (generative QA) stays at 57.9, and ChineseQA drops slightly from 75.0 to 74.4. The paper's conclusion is that including MC data inflates multiple-choice benchmark scores without improving genuine understanding or generation capability: "users may not perceive the model as becoming more intelligent during conversational interactions." This finding directly informs the paper's methodological choice to exclude MC data from both pre-training and fine-tuning, a deliberate decision to avoid "benchmark decoration." The magnitude of the inflation (10–24 points across major benchmarks) is striking and serves as a caution for the broader community about interpreting benchmark scores from models that may have seen multiple-choice data during training.

**Instruction Data in Pre-Training (Section 5.5):** An experiment incorporating 5 million instruction data instances (primarily multiple-choice questions) during the final 10% of pre-training showed improved benchmark performance, but "the final outcomes were nearly identical to those achieved by adding the same data during the SFT stage." This means that mixing instruction data into pre-training provides no advantage over simply including it in the fine-tuning phase. The paper's decision to exclude instruction data from pre-training is based on both this equivalence finding and a preference for avoiding multiple-choice data in general: "Due to our preference for excluding multi-choice questions and the limited availability of non-multi-choice questions we have, we made the decision not to include instruction data in the pre-training process." This is a pragmatic engineering choice documented with empirical justification, not just an assertion.

**System Prompt Effect by Model Scale (Section 5.5, Table 14):** The system prompt produces opposite effects at different scales: for the 7B Chat model, adding the system prompt decreases MT-Bench from 7.15 to 7.11 (−0.04), while for the 67B Chat model, it increases MT-Bench from 8.35 to 8.58 (+0.23). This is a non-obvious interaction: the same system prompt is helpful to the larger model but slightly harmful to the smaller one. The paper's explanation — "larger models possess a better understanding of the intended meaning behind the system prompt" while "smaller models struggle to grasp the system prompt adequately, and the inconsistency between training and testing might negatively impact their performance" — is plausible but not empirically verified (no ablation on system prompt wording or training to understand system prompts is provided). The result serves as a robustness check on the 67B Chat model's open-ended evaluation: the system prompt improves performance, so the strong AlignBench and MT-Bench results are not an artifact of prompt engineering that only works at one scale.

**Scaling Law Exponents Across Different Datasets (Section 3.3, Table 4):** The model scaling exponent $a$ varies systematically from 0.450 (early in-house data) to 0.524 (current in-house data) to 0.578 (OpenWebText2). The data scaling exponent $b$ correspondingly decreases from 0.550 to 0.476 to 0.422. This demonstrates that the optimal allocation strategy is not a universal constant — it depends on data quality. The robustness here is that the three datasets span different quality levels and consistently show the same directional trend: higher quality → larger $a$ (bias toward model scaling). The paper does not provide an independent quantitative measure of data quality (the quality ranking is based on the team's "internal data assessment"), which means the correlation is qualitative rather than quantitatively established. However, the finding is internally consistent and provides a plausible explanation for the Kaplan-Hoffmann discrepancy.

**Model Scale Representation Comparison (Appendix A.2, Figure 6):** The paper refits the performance scaling curve using three different model scale representations: $6N_1$ (non-embedding parameters, Kaplan's metric), $6N_2$ (complete parameters, Hoffmann's metric), and $M$ (non-embedding FLOPs/token). The results show that $6N_1$ overestimates the performance of large models (the 7B and 67B stars fall below the fitted line — actual performance is better than predicted), while $6N_2$ underestimates performance (stars fall above the fitted line — actual performance is worse than predicted). Only $M$ produces accurate predictions where the stars lie on the fitted curve. This is a critical robustness check on the paper's methodological contribution: the $M$ representation is validated by its predictive accuracy at 1000× the fitting budget, while the alternatives produce systematic prediction errors that would lead to suboptimal scaling decisions.

**DPO Training Epoch Count (Section 4):** The paper trains DPO for exactly 1 epoch with a learning rate of 5×10⁻⁶ and batch size 512. No ablation on epoch count or learning rate is reported. The justification is indirect: Table 17 shows minimal benchmark degradation after DPO, suggesting 1 epoch is sufficient without causing catastrophic forgetting. However, without an ablation comparing 1 vs. 2 vs. 3 epochs of DPO, it's impossible to know whether the open-ended generation improvements (AlignBench and MT-Bench gains) could have been larger with more DPO training, or whether benchmark degradation would eventually set in.

**SFT Epoch Count and Model Scale Interaction (Section 4):** The 7B model is fine-tuned for 4 epochs while the 67B model gets only 2 epochs. The paper explains: "we observed the overfitting problem is serious on the 67B model" and "the 67B model hits the upper bound soon" on GSM8K and HumanEval. This is a practical robustness finding: larger models require fewer SFT epochs before overfitting. No formal overfitting metric (e.g., validation loss divergence) is reported, so the epoch count decision appears to be based on monitoring benchmark scores during training rather than a principled early-stopping criterion.

**Repetition Ratio Monitoring (Section 4):** The paper introduces a repetition ratio metric — "the proportion of generated responses that fail to terminate and instead endlessly repeat a sequence of text" — computed on a set of 3868 Chinese and English prompts. This metric is used to diagnose SFT quality and motivate the two-stage fine-tuning approach. No ablation is presented on how the repetition ratio varies with decoding temperature, SFT data composition, or model scale beyond the observations already discussed. The metric serves as a practical diagnostic, but its relationship to user-perceived quality or benchmark performance is not systematically explored.

### Critical Assessment

#### Does the paper demonstrate that DeepSeek LLM 67B "surpasses LLaMA-2 70B across a range of benchmarks"?

**Yes, with an important qualification about matched vs. unmatched pre-training data.** Table 5 shows consistent advantages for DeepSeek 67B over LLaMA-2 70B, particularly in code (HumanEval: 42.7 vs. 28.7), math (MATH: 18.7 vs. 13.5, GSM8K: 63.4 vs. 58.4), and Chinese tasks (CHID: 92.1 vs. 55.5, C-Eval: 66.1 vs. 51.4). On English language understanding, the models are roughly comparable (HellaSwag tied at 84.0, MMLU 71.3 vs. 69.0, PIQA 83.6 vs. 82.0), with DeepSeek leading on some and LLaMA-2 on others. The reported gaps are consistent and mostly outside the range of run-to-run variance for benchmarks of this size, though the paper doesn't provide error bars.

However, the comparison is not entirely apples-to-apples. LLaMA-2 was trained on 2T tokens of primarily English data, while DeepSeek was trained on 2T tokens of bilingual data. DeepSeek's advantages on Chinese benchmarks are expected and reflect the language composition of the training data, not necessarily a better training methodology. DeepSeek's advantages on code and math (which are largely language-agnostic) are more significant because they suggest genuine capability improvements independent of data language composition. The paper acknowledges this implicitly by noting that LLaMA-2's performance on CMath (53.9%) despite no Chinese training suggests math reasoning transfers across languages — yet DeepSeek still substantially outperforms it (63.0%).

The comparison at 7B scale is less decisive: DeepSeek leads on code and Chinese but the gaps are smaller, and on some English tasks LLaMA-2 7B actually outperforms DeepSeek 7B (e.g., TriviaQA: 63.8 vs. 59.7, NaturalQuestions: 25.5 vs. 22.2, ARC-Challenge: 49.0 vs. 48.1). This is consistent with the paper's observation about "language conflict" in smaller models, but it also means the superiority claim is less robust at 7B — the model is better at some things and worse at others, not uniformly better.

**What would have strengthened this claim:** (1) A comparison against LLaMA-2 trained on the same bilingual data mixture would isolate the effect of the DeepSeek training methodology from the effect of data language composition. Such a comparison is obviously infeasible (you'd need to retrain LLaMA-2), but it's worth noting that the current comparison conflates architecture/training choices with data composition choices. (2) Error bars or statistical significance tests on the benchmark comparisons would help identify which gaps are reliable vs. within noise. (3) More challenging coding benchmarks beyond HumanEval and MBPP (which are becoming saturated) would better differentiate the models' coding capabilities.

#### Does the scaling law framework accurately predict 7B/67B performance from small-scale experiments?

**The evidence is positive but limited.** Figure 5 shows the performance scaling curve where the 7B and 67B models (blue stars) lie on the fitted line extrapolated from models with up to 1000× smaller compute budgets. This is visually compelling and the paper interprets it as validation. However, the validation is limited to exactly two data points (7B and 67B) — the paper does not test predictions at intermediate scales (e.g., 13B, 34B) to confirm that the fitted curve is well-calibrated across the entire range rather than just at the two tested points coincidentally landing on the curve. A stronger validation would involve withholding one or more of the IsoFLOP budget levels during fitting and checking whether the held-out points lie on the predicted curve.

The validation metric is validation loss (bits-per-byte), not downstream task performance. The scaling law predicts that validation loss decreases with compute, but the relationship between validation loss and benchmark accuracy is not modeled. So the claim "scaling laws predict performance" is true only for next-token prediction loss, not for the benchmark scores that matter to users. The paper implicitly acknowledges this limitation by not claiming that benchmark scores can be predicted — only that the validation loss at 7B and 67B matches predictions.

**A key gap:** The paper validates the performance scaling curve (Figure 5) but does not validate the optimal allocation predictions per se. Specifically, it does not test whether training a model at a non-optimal (M, D) allocation and comparing its loss to a model at the optimal allocation matches what the IsoFLOP curves predict. The 7B and 67B models were presumably trained at approximately the optimal allocation for their scale, but this is never explicitly verified against a counterfactual. If the 67B model had been trained on 1T tokens instead of 2T, would its validation loss have been higher by the amount the scaling law predicts? Without this kind of ablation, the optimal allocation claim remains an extrapolation from small-scale experiments rather than a verified prediction.

#### Does the paper demonstrate that higher data quality shifts the optimal allocation toward model scaling?

**The evidence is consistent but correlational, not causal.** Table 4 shows three datasets with different quality levels and different fitted exponents. The quality ranking (early < current < OpenWebText2) is based on the authors' assessment, not on an independent quantitative metric. The paper states that "the quality of OpenWebText2 even surpasses the current in-house data, due to its smaller scale which allows for more meticulous processing" — but this is a qualitative judgment, not a measured property. To make this claim causal, the paper would need to manipulate data quality while holding other factors constant (e.g., create multiple versions of the same dataset with varying quality levels through controlled filtering or noise injection) and measure how the exponents shift. What the paper actually shows is that three different datasets, which the authors believe differ in quality, produce different exponents. The monotonic relationship with quality is suggestive but could be confounded by other dataset properties (domain composition, language mix, tokenizer, etc.).

The explanatory mechanism offered — "high-quality data usually implies logical clarity and less predictive difficulty after sufficient training" — is intuitive but untested. How would one verify that "less predictive difficulty" is the mechanism? One approach would be to measure the entropy or compressibility of each dataset and correlate it with the fitted exponents. The paper does not do this.

That said, the finding is robust in a weaker sense: the exponents are demonstrably non-universal, which is sufficient to establish that practitioners should not blindly adopt Kaplan's or Hoffmann's exponents for their own training runs. The paper need not establish a precise causal mechanism for the claim "you should fit exponents on your own data" to be valid and important. The contribution is establishing that doing so matters, not explaining exactly why.

#### Does the DeepSeek 67B Chat model genuinely "outperform GPT-3.5 in open-ended evaluations"?

**Yes, with evidence from two independent benchmarks.** On AlignBench (Chinese, Table 7), DeepSeek 67B Chat DPO scores 6.69 overall vs. GPT-3.5-turbo-0613 at 6.08 — a 0.61 point advantage. On MT-Bench (English, Table 8), DeepSeek 67B Chat DPO scores 8.76 vs. GPT-3.5-turbo-0613 at 8.39 — a 0.37 point advantage. The SFT-only model (without DPO) scores 8.35 on MT-Bench, essentially tied with GPT-3.5 (8.39) and still within "outperform" territory on AlignBench (6.43 vs. 6.08).

**Caveats on the comparison:**

1. **Evaluator bias.** Both AlignBench and MT-Bench use GPT-4 as the judge. There is a known concern that GPT-4 may exhibit systematic biases in evaluating other models' outputs (e.g., position bias, verbosity bias, self-enhancement bias). The paper uses the official evaluation code from both benchmarks, but it does not report any calibration of the judge against human preferences. If GPT-4 exhibits any systematic bias against GPT-3.5 outputs or in favor of certain open-source model outputs, the comparison could be skewed.

2. **Single prompt per question.** MT-Bench generates one response per question per model (using a single temperature setting) and evaluates it once. For open-ended generation, single-sample evaluation has high variance — a different random seed could produce a meaningfully different score. The paper does not report standard deviations or multiple-evaluation runs.

3. **GPT-3.5 version.** The comparison is against GPT-3.5-turbo-0613, which was the standard version at the time of evaluation but is not the most capable GPT-3.5 variant. Different GPT-3.5 versions have different performance profiles, and the paper does not clarify whether this specific version was chosen for comparability or was simply the most recent at the time.

4. **Category-level variation.** DeepSeek outperforms GPT-3.5 on Reasoning (MT-Bench 9.05 vs. 6.20) but underperforms on Coding (6.75 vs. 7.05) and Math (6.65 vs. 7.05) in the DPO variant. The overall average advantage is driven by large gains in specific categories rather than uniform superiority. The SFT-only model shows the opposite pattern — stronger on Coding (7.35) but weaker on Reasoning (8.00). So "outperforms GPT-3.5" is an average statement that masks category-level heterogeneity, and which model "wins" depends on the task.

5. **The DPO model is post-alignment, while GPT-3.5 is RLHF-trained.** The comparison is between DeepSeek after SFT + DPO and GPT-3.5 after RLHF. The alignment pipelines are not equivalent, and it's unclear whether the advantage comes from better pre-training, better SFT data, or DPO's superiority over RLHF for this evaluation. An ablation comparing DeepSeek Chat SFT (without DPO) to GPT-3.5 would isolate the effect of the base model + SFT pipeline.

#### Does the safety evaluation show genuine safety, or is it measuring primarily refusal behavior?

The Do-Not-Answer score (Table 11) rewards models for refusing to answer unsafe queries. The ranking shows LLaMA-2-7B-Chat at 99.4 (the highest), Claude at 98.3, and DeepSeek at 97.8. The very high score of LLaMA-2-7B-Chat, a model known in the community for being overly cautious and refusing many benign queries, suggests the metric may over-reward refusal behavior at the expense of helpfulness. DeepSeek's score of 97.8 being only 0.1 points above ChatGPT (97.7) makes this a statistical tie. The paper claims it "places it amongst the ranks of the safest models," which is technically true but overstates a difference of 0.1 points on a 100-point scale.

The in-house safety evaluation (Table 10) is more informative but has limitations. The 2400 questions were constructed by the DeepSeek team's own experts, which introduces potential for benchmark design that favors the model. The safety rate of ~95.8% is reported without comparison to any baseline models, so there is no way to assess whether this is good or bad relative to alternatives. The annotation process (three-category annotation with cross-verification) is well-described, but inter-annotator agreement metrics are not reported.

**Key missing evaluation:** The paper does not evaluate the trade-off between safety and helpfulness. A model that refuses all queries is perfectly safe but useless. The safety evaluation would be strengthened by measuring over-refusal rates on benign queries (e.g., the XSTest or similar benchmarks) to quantify how much helpfulness is sacrificed for safety. Without this, the safety numbers are only meaningful in the context of how the model performs on legitimate requests.

#### Are the held-out evaluation results sufficient to demonstrate genuine capability vs. benchmark overfitting?

**The held-out evaluations are a strength of the paper and provide meaningful signal, but the interpretation requires care.** Table 9 shows consistent drops from standard benchmarks to held-out benchmarks: ChatGLM3 goes from 52.4 on MBPP to 2.4 on LeetCode; GSM8K scores don't predict Hungarian Exam scores well. DeepSeek 67B leads open-source models on all three held-out benchmarks, which is consistent with the claim that its strong standard benchmark performance reflects genuine capability rather than overfitting.

However, the held-out benchmarks have their own limitations. LeetCode (126 problems) is a relatively small sample from one contest format. The Hungarian Exam (33 problems) is even smaller, and human annotation introduces subjectivity (inter-annotator agreement is not reported). IFEval (500 prompts) is adequately sized, but the "prompt-level loose" metric may be forgiving of partial instruction-following failures. None of these benchmarks have been widely adopted or validated in the community, which limits their interpretability — a score of 55.5 on IFEval is hard to calibrate without knowing how scores correlate with real-world instruction-following quality.

More critically, the paper does not provide a direct comparison of standard vs. held-out benchmark rankings across all models. The informal observation that "ChatGLM3 achieves a score of 52.4 on MBPP... but when evaluated on new benchmarks, its performance falls considerably short" is anecdotal. A systematic analysis showing the rank correlation between standard and held-out benchmarks for all evaluated models would quantify how much benchmark overfitting is occurring and whether DeepSeek is an outlier in either direction.

#### What experiments are missing?

1. **Scaling law validation at intermediate scales.** The paper predicts 7B and 67B performance from experiments at 1000× smaller budgets, but doesn't validate at 1B, 3B, or 13B scales to confirm the fitted curve is accurate across the entire range, not just at the two endpoints.

2. **Ablation on the $M$ representation against a held-out allocation counterfactual.** The paper claims $M$ is superior to $N_1$ and $N_2$, but doesn't test whether using $N_1$ or $N_2$ to guide allocation decisions would have produced a measurably worse 67B model. The validation in Appendix A.2 only checks whether the predicted validation loss matches — it doesn't check whether following the wrong allocation advice leads to suboptimal models in practice.

3. **Data quality manipulation experiment.** To establish that data quality *causes* the exponent shift rather than merely correlating with it, the paper could take one dataset and create variants at different quality levels (through filtering thresholds, noise injection, or data source selection) and measure how the exponents change. Without this, the causal claim remains speculative.

4. **Safety-helpfulness trade-off measurement.** Evaluations on over-refusal benchmarks would contextualize the safety scores and reveal whether the model is genuinely safe or simply unhelpfully cautious.

5. **DPO ablation on data scale and epoch count.** The paper trains DPO for 1 epoch with a fixed dataset. Understanding how DPO performance varies with more data, more epochs, or different learning rates would help practitioners replicate the alignment pipeline.

6. **Comparison against a stronger pretrained baseline.** The paper compares against LLaMA-2 but not against other contemporaneous models like Mistral 7B, Qwen 72B base, or Yi-34B base. Including these comparisons would better situate DeepSeek's performance in the landscape.

7. **Error analysis on benchmark failures.** Beyond aggregate accuracy, understanding *where* the model fails (e.g., does MATH performance degrade on geometry vs. algebra? Does HumanEval failure correlate with problem length?) would provide actionable insight for future improvement.

#### Overall Assessment of the Experimental Support

The paper's experimental evaluation is **broad and thorough in coverage**, spanning standard benchmarks, open-ended evaluations, held-out tests, and safety assessments across both English and Chinese. The key performance claims — DeepSeek 67B surpasses LLaMA-2 70B on code/math/reasoning, DeepSeek 67B Chat outperforms GPT-3.5 on open-ended generation — are supported by multiple independent evaluations with consistent results.

The scaling law claims are more nuanced. The paper demonstrates that its fitted scaling laws accurately predict validation loss at 7B and 67B scales (a genuine achievement), but the evidence that the optimal allocation strategy would produce better models than alternative strategies is indirect — it rests on the fitted exponents being correct, which is validated only through the two-point performance prediction, not through a head-to-head comparison of models trained at optimal vs. suboptimal allocations. The data quality finding is empirically robust (different datasets produce different exponents) but causally incomplete (the mechanism is speculative).

The methodological contributions — the $M$ representation, hyperparameter scaling laws, avoidance of benchmark decoration — are well-documented and supported by the experiments that are present. The decision to exclude multiple-choice data from training and to report the inflation it would cause is a notable example of experimental rigor that prioritizes validity over benchmark scores.

The main weaknesses are: (1) the absence of statistical uncertainty quantification for nearly all results, making it impossible to assess which gaps are reliable; (2) the relatively small size of the held-out and safety test sets, which limits the precision of those evaluations; (3) the lack of ablation experiments on key design choices in the alignment pipeline (DPO epochs, SFT data composition effects beyond the two-stage comparison, system prompt wording); and (4) the indirect nature of the scaling law validation, which shows predictive accuracy but doesn't demonstrate that following the scaling law advice yields better models than alternative strategies would. These weaknesses do not undermine the paper's core claims, but they leave open questions that future work should address.

## 6. Limitations and Trade-offs

### 6.1 The Scaling Law Validation Relies on Only Two Large-Scale Data Points

**The assumption or constraint.** The paper's central claim is that its scaling law framework — hyperparameter power laws, the $M$ representation, and the IsoFLOP-based optimal allocation — accurately predicts the performance of models at 1000× the compute budget of the fitting experiments. The validation for this claim comes from exactly two models: DeepSeek 7B and DeepSeek 67B, whose validation loss (bits-per-byte) lies on the extrapolated scaling curve in Figure 5. The paper treats this as confirmation:

> "Using small-scale experiments can accurately predict the performance of models with 1000× compute budget. This provides both confidence and guidance for training models on a larger scale."

**The consequence.** Two data points cannot distinguish between a well-fitted scaling law and a regression line that happens to intersect the true values at those two scales by coincidence. The fitted curve is almost entirely anchored by models at compute budgets below $2\times10^{19}$ FLOPs — roughly the scale of a 1–3B parameter model. The 7B and 67B models sit at compute budgets of approximately $2\times10^{21}$ and $2\times10^{22}$ FLOPs respectively (roughly 100× and 1000× beyond the fitting range, depending on the specific $M$ value). If the scaling law's functional form deviates from a pure power law at intermediate scales — for example, if there is a phase transition or saturation effect between 3B and 7B parameters — it would not be detectable from the current validation. The paper would need predictions at intermediate scales (1B, 3B, 13B, 34B) to confirm that the curve holds across the entire range rather than just at its endpoints coincidentally landing on the fitted line.

This matters because the scaling law is presented as the key decision-making tool for future model development. If the predictions are actually less reliable than they appear from the two-point validation, the team risks making multi-million-dollar scaling decisions based on an overfitted extrapolation.

**What evidence exists in the paper.** Figure 5 shows the two data points. Appendix A.2 (Figure 6) repeats the analysis with different model scale representations ($6N_1$, $6N_2$, $M$) and shows that only $M$ produces predictions where the stars lie on the fitted line. The fact that $6N_1$ and $6N_2$ produce biased predictions (stars systematically above or below the curve) is evidence that the $M$ representation is *better*, but it does not increase the effective sample size. The validation remains two points at the high end.

**Mitigation status.** The paper does not acknowledge this as a limitation. Section 6 frames the scaling law findings as successfully validated and lists future work about "a larger and improved dataset for the upcoming version of DeepSeek LLM," but does not discuss the need for validation at intermediate scales. The confidence expressed ("this provides both confidence and guidance") is stronger than the evidence warrants.

---

### 6.2 The Data Quality–Scaling Exponent Relationship Is Correlational, Not Causal, and Lacks Independent Quality Metrics

**The assumption or constraint.** Section 3.3 reports that the optimal model/data scaling-up allocation strategy shifts when using datasets of different quality: higher-quality data produces a larger model scaling exponent $a$ and a smaller data scaling exponent $b$. The paper treats data quality as the causal factor:

> "The higher the data quality, the more the increased compute budget should be allocated to model scaling."

However, the quality ranking of the three datasets (early in-house < current in-house < OpenWebText2) is based on the authors' internal assessment, not on an independent quantitative metric:

> "Our internal data assessment revealed that current in-house data has higher data quality than early in-house data. Furthermore, the quality of OpenWebText2 even surpasses the current in-house data, due to its smaller scale which allows for more meticulous processing."

**The consequence.** The datasets differ on multiple dimensions simultaneously: language composition (in-house is bilingual Chinese-English, OpenWebText2 is English-only), domain distribution, date range, deduplication procedures, and filtering criteria. Any of these factors — or an interaction between them — could drive the observed exponent differences independently of the "quality" construct as the authors define it. A practitioner reading this paper might conclude that investing in data curation will predictably shift their optimal allocation toward larger models. But the paper provides no mechanism for *how much* the exponents will shift for a given quality improvement, nor any way to predict the exponents from measurable dataset properties (entropy, token diversity, factual density) without running their own full IsoFLOP experiments. Without a causal link, the paper's recommendation to "fit the exponents on your own data" is the only certain takeaway — but this weakens the value of the paper's specific fitted exponents as a reference for other practitioners.

**What evidence exists in the paper.** Table 4 reports the fitted exponents across the three datasets, and the shift is consistent in direction. However, the paper does not report any quantitative metric of data quality (e.g., training loss on a reference model, perplexity, deduplication rate, information density) for any of the three datasets. The quality ranking is entirely qualitative. The paper acknowledges this indirectly:

> "We will continue to pay close attention to the changes in data quality and its impact on scaling laws, and provide more analysis in future works."

This is a transparent acknowledgment that the analysis is incomplete, but it does not correct the causal implication in the paper's main claims.

**Mitigation status.** The limitation is partially acknowledged through the future work statement. However, the paper does not qualify its causal language when presenting the finding as a key contribution. The claim that data quality "significantly influences the optimal model/data scaling-up allocation strategy" is stated as a conclusion, not as a hypothesis requiring further validation.

---

### 6.3 The $14\times$ Larger Pretraining Baseline Is Not Compute-Optimally Configured for a Fair Trade-Off Comparison

**The scaling law trade-off comparison against pretraining is defined but not empirically tested.** This requires clarification: the DeepSeek paper does NOT conduct a FLOPs-matched comparison between a smaller DeepSeek model with test-time compute and a larger DeepSeek model with greedy decoding. There is no pretraining-vs-inference trade-off experiment anywhere in Section 3. The scaling laws are about *pretraining* resource allocation only — how to split a fixed pretraining budget between model size $M$ and data quantity $D$. The "optimal" in the paper is always optimal *pretraining* allocation, not a trade-off between pretraining and some alternative compute usage.

However, the paper's model performance claims **rest on a baseline comparison against LLaMA-2 70B** that is not fully controlled for training data composition, and the scaling law predictions **are validated on models whose architectures differ from the small-scale fitting experiments** in ways that are not fully accounted for. The small-scale experiments in Section 3 are run on models with a range of layer counts and hidden dimensions, but they all share the same architecture template (Pre-Norm, SwiGLU, RoPE, etc.). The 7B and 67B models use the same template but with different specific choices — the 67B model uses GQA while the small-scale models presumably use MHA, and the depth-to-width ratios are not held constant (the 67B model is deeper than LLaMA-2 70B, with 95 vs. 80 layers). These architectural variations interact with scaling behavior in ways the paper does not model.

**The consequence.** If the architecture of the large models differs from the small-scale fitting models in ways that systematically affect compute efficiency — for example, if GQA changes the relationship between FLOPs/token and model capacity — then the extrapolated predictions from Figure 5 are not purely about scale. They conflate scaling effects with architectural effects. A practitioner attempting to use these scaling laws to predict the performance of a model with a different attention mechanism, layer count, or FFN configuration might get predictions that are systematically off, because the scaling laws are fitted on one architectural family and validated on a slightly different one.

**What evidence exists in the paper.** Section 2.2 specifies the architecture differences: the 67B model uses GQA with 8 KV heads while the 7B model uses standard MHA, and the 67B model "expanded parameters in network depth rather than the common practice of widening the intermediate width of FFN layers." The paper does not discuss whether the small-scale IsoFLOP experiments also used GQA or varied depth-to-width ratios in the same way. Table 2 shows that the 67B model has 95 layers vs. 80 for LLaMA-2 70B — a deliberate architectural choice — but the small-scale fitting experiments' architectures are not described in comparable detail.

**Mitigation status.** Not addressed. The paper presents the scaling law predictions as architecture-independent ("our findings that facilitate the scaling of large scale models"), but the architectural choices made in the 7B and 67B models are not explicitly connected back to the small-scale experiments' architectures. This is a gap in the experimental control that weakens the claim that the scaling laws alone account for the performance achieved.

---

### 6.4 Difficulty Estimation Cost Is Zero in the Headline Numbers — But Is Prohibitively Expensive in Practice

**The assumption or constraint.** The scaling laws presented in Section 3 require fitting power-law exponents using IsoFLOP experiments — training approximately 10 different model/data configurations at each of 8 different compute budgets, plus hyperparameter grid searches at each budget. This fitting process is enormously expensive: it involves training dozens of models at scales up to roughly $3\times10^{20}$ FLOPs before the large-scale models are even started. The paper treats this cost as a sunk research investment and does not include it in any of the reported "compute-optimal" figures. The 7B and 67B models are then claimed to be near-optimal based on the fitted exponents, but the total cost of *discovering* those exponents is not factored into any efficiency calculation.

The paper is transparent that the scaling law experiments were conducted, but the framing treats them as a one-time calibration effort rather than an ongoing cost:

> "We selected 8 different compute budgets ranging from 1e17 to 3e20, and designed around 10 different model/data scale allocations for each budget."

For a practitioner adopting this methodology, the cost of fitting scaling laws on their own data mixture and infrastructure would be substantial — likely tens of thousands of GPU-hours — and this cost must be amortized over however many models they plan to train.

**The consequence.** The paper's framing — "guided by the scaling laws, we introduce DeepSeek LLM" — can be read as implying that the scaling laws *saved* compute by preventing suboptimal allocation choices. This is likely true for a long-term project training many models (the "longtermism" thesis), but it is not quantified. A team planning to train a single 7B or 67B model would not recoup the scaling law fitting cost. The headline claim that scaling laws enable "efficient scale-up" is thus conditional on the scale of ambition: it is efficient only if you plan to train many future models using the same infrastructure, data pipeline, and architectural template, such that the calibration cost can be amortized.

Additionally, practitioners with different data mixtures must re-fit the exponents (per the data quality finding in Section 3.3), meaning the cost is not a one-time payment for the community — it is a per-team, per-dataset cost that limits the portability of the paper's specific fitted numbers.

**What evidence exists in the paper.** The paper does not report the total compute spent on scaling law experiments. Section 3 describes the experimental design but gives no FLOPs budget for the calibration process itself. The computational cost can be approximately inferred: 8 budgets × ~10 allocations = ~80 models trained, plus hyperparameter grid searches at multiple budgets. Many of these models are small (the lowest budget is $10^{17}$ FLOPs, roughly equivalent to training a model with ~20M non-embedding parameters on ~150M tokens), but the highest budget of $3\times10^{20}$ FLOPs is substantial. The paper does not account for this cost anywhere.

**Mitigation status.** Not addressed. The paper frames the scaling law findings as a contribution that benefits the community by enabling better scaling decisions, but the cost of producing those findings — and the cost of replicating them — is not discussed. This is a practical limitation for anyone who wants to adopt the methodology rather than just use the already-trained DeepSeek models.

---

### 6.5 Single Benchmark Domain (MATH) and Single Model Family (PaLM 2-S*) — All Results Are Conditional on These Choices

Wait — I need to correct myself. The DeepSeek paper does NOT use the MATH benchmark as its primary evaluation and does NOT use PaLM 2-S*. Let me review what the paper actually uses.

The DeepSeek paper evaluates on a broad suite of benchmarks including MMLU, GSM8K, MATH, HumanEval, MBPP, BBH, HellaSwag, TriviaQA, etc. (Section 5.1). It is NOT limited to MATH. And the model family is DeepSeek LLM itself (7B and 67B), not PaLM. The scaling law experiments use smaller variants of the same architecture.

The relevant limitation for this paper is different — it's about the **scaling law experiments being conducted on a single architectural family and a single data distribution** (the DeepSeek in-house data pipeline), with no validation on different architectures or data mixtures beyond the three datasets compared in Table 4.

**The assumption or constraint.** All scaling law experiments in Section 3 are conducted within the DeepSeek LLM architectural family: Pre-Norm transformers with RMSNorm, SwiGLU activations, RoPE positional encoding, and the specific FFN ratio of $\frac{8}{3}d_{\text{model}}$. The fitted exponents ($a = 0.5243$, $b = 0.4757$ for the current dataset) and hyperparameter scaling relationships ($\eta_{\text{opt}} \propto C^{-0.1250}$, $B_{\text{opt}} \propto C^{0.3271}$) are conditional on this architecture, this optimizer (AdamW with specific $\beta$ values and weight decay), and this data pipeline. The paper states this implicitly by fitting the exponents on its own experiments, but does not discuss whether different architectural choices (e.g., different normalization schemes, activation functions, attention mechanisms, or depth-to-width ratios) would produce different scaling exponents.

**The consequence.** A practitioner using a different architecture — for instance, a model with multi-query attention instead of GQA, or a different FFN design, or a different positional encoding — cannot assume the paper's fitted exponents will transfer. Even within the DeepSeek family, the 67B model uses GQA while the small-scale fitting experiments may not (the paper does not specify), introducing an unmodeled architectural confound. The paper's specific numerical predictions (Equation 4) are only directly applicable to models that closely match the DeepSeek architectural template. The qualitative findings — hyperparameter scaling laws exist, data quality matters, $M$ is better than $N_1$ or $N_2$ — are more portable than the specific fitted constants, but the paper does not distinguish between portable qualitative insights and architecture-specific quantitative predictions.

**What evidence exists in the paper.** The paper does not conduct any cross-architecture scaling law comparisons. The three datasets compared in Table 4 all use the same underlying architecture. There is no ablation where, for example, a model with MHA and a model with GQA are fitted with separate scaling laws to see if the exponents differ. The paper acknowledges one architectural choice that affects scaling — expanding the 67B model in depth rather than width — but does not connect this to the scaling law fitting process.

**Mitigation status.** Not addressed. The paper presents the scaling laws as general findings ("we delve into the study of scaling laws and present our distinctive findings that facilitate the scaling of large scale models") without discussing the architectural conditionality. The title's reference to "Scaling Open-Source Language Models" implies generality that the experiments do not fully support.

---

### 6.6 The Alignment Pipeline Evaluation Lacks Direct Comparisons Against Standard Alignment Baselines (RLHF) and Does Not Quantify the Safety–Helpfulness Trade-off

**The assumption or constraint.** The paper's alignment pipeline uses SFT followed by DPO, and evaluates the resulting chat models against GPT-3.5, GPT-4, and various open-source chat models. However, the paper does not include an **RLHF baseline** (e.g., PPO-based training with a learned reward model) to compare against DPO, despite RLHF being the dominant alignment paradigm used by the closed-source models (GPT-3.5, GPT-4) that DeepSeek Chat is compared against. The paper justifies DPO as "a simple but effective method for LLM alignment" (Section 4), but provides no evidence that DPO specifically — rather than alignment in general — is responsible for the observed open-ended generation improvements.

Additionally, the safety evaluation (Section 5.4) reports safety scores without measuring the associated helpfulness degradation. The Do-Not-Answer score of 97.8 (Table 11) and the in-house safety rates of ~95.8% (Table 10) are presented without any measurement of how often the model refuses benign queries or degrades its response quality on legitimate requests. A model that achieves high safety by refusing to engage with a broad set of queries is not necessarily preferable to a model that is slightly less safe but substantially more helpful.

**The consequence.** For the comparison against GPT-3.5: if DPO is less effective than RLHF for alignment, then DeepSeek Chat's ability to approach or exceed GPT-3.5 on open-ended benchmarks (MT-Bench 8.35–8.76 vs. GPT-3.5's 8.39, AlignBench 6.43–6.69 vs. GPT-3.5's 6.08) might indicate that the underlying base model is *substantially stronger* than GPT-3.5's base model, and the alignment gap is being overcome by base capability rather than alignment quality. Conversely, if DPO is more effective than RLHF, the alignment advantage might not transfer to architectures or data scales where DPO is less suitable. Without an RLHF baseline, the source of the performance is ambiguous.

For the safety evaluation: the paper cannot claim that DeepSeek Chat is "safe" in a practically useful sense without also showing that safety does not come at an unacceptable cost to helpfulness. The LLaMA-2-7B-Chat model scores 99.4 on Do-Not-Answer (higher than DeepSeek's 97.8), yet LLaMA-2-7B-Chat is widely recognized in the community as excessively cautious. DeepSeek's score being close to this model raises the question of whether it exhibits similar over-refusal behavior, but the paper provides no data to assess this.

**What evidence exists in the paper.** Table 17 shows that DPO causes minimal changes on standard benchmarks (MMLU: 71.1 → 70.9, GSM8K: 84.1 → 85.2, MATH: 32.6 → 30.2), confirming that DPO does not catastrophically degrade base capabilities. Tables 7 and 8 show that DPO improves open-ended evaluation scores. But no RLHF-trained DeepSeek model exists to compare against, and no over-refusal evaluation is reported. The DPO vs. RLHF question is entirely unaddressed.

**Mitigation status.** The paper does not acknowledge the absence of an RLHF baseline as a limitation. Section 4 states that DPO was used because it "is proven to be a simple but effective method" — a pragmatic justification, not a comparative claim. The safety evaluation in Section 5.4 does not mention the safety–helpfulness trade-off or the risk of over-refusal. Section 6 (Conclusion) lists "the possibility of generating non-factual information such as unverified advice, and a tendency to produce hallucinations" as limitations, but does not discuss over-refusal or safety–helpfulness calibration.

## 7. Implications and Future Directions
- Field impact
  - The compute identity `C = M · D` with `M` as non‑embedding FLOPs/token and the fitted exponents (Eq. (4)) provide a practical playbook for future open‑source LLM training. Researchers can plan budgets and predict performance (Figure 5) rather than guess. The observation that better data favors larger models helps reconcile Kaplan‑ vs. Chinchilla‑style guidance (Table 4).
- Follow‑up research
  - Generalize the compute model to other architectures (e.g., Mixture‑of‑Experts, retrieval‑augmented, long‑context attention) and verify whether analogous `M` definitions lead to stable exponents.
  - Formalize “data quality” with measurable proxies (e.g., perplexity filtering, diversity/novelty metrics) and test how each aspect shifts the `a/b` exponents.
  - Extend hyperparameter scaling to include weight decay, warmup schedules, and optimizer variants; analyze sensitivity to `(M, D)` composition (Section 3.1 notes residual dependence).
  - Explore reinforcement learning–based alignment to further improve complex reasoning (Section 6 “Conclusion” hints at positive early results).
- Practical applications
  - The 67B chat model’s strength on math/code (Tables 6, 8, 9, A.4) suggests immediate use in programming assistants, math tutoring, and enterprise Q&A in bilingual (Chinese/English) settings.
  - The strong safety performance (Tables 10–11) and two‑stage SFT recipe (Table 12) offer concrete guidelines for deploying helpful, low‑repetition chat systems without overfitting to multiple‑choice formats (Table 13).

> In short, DeepSeek LLM contributes a tested methodology for compute‑optimal scaling—hyperparameters, model/data allocation, and alignment choices—and validates it with competitive 7B/67B bilingual models trained on 2T tokens. The work offers both actionable recipes (Eq. (1), Eq. (4)) and conceptual insights (data quality’s role) that future open‑source LLM projects can adopt and extend.
