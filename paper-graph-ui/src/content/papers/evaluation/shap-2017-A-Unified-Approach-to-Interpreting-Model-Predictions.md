# A Unified Approach to Interpreting Model Predictions

**URL:** [https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)

## 🎯 Pitch

This paper introduces a unified framework for interpreting individual model predictions, **SHAP (SHapley Additive exPlanation) values**, which assigns each input feature an importance measure for a specific prediction.

---

## 1. Executive Summary

This paper introduces a unified framework for interpreting individual model predictions, **SHAP (SHapley Additive exPlanation) values**, which assigns each input feature an importance measure for a specific prediction. By viewing any explanation as an additive feature attribution model — a linear function of binary variables indicating feature presence — the authors prove that Shapley values from cooperative game theory provide the unique solution satisfying three desirable properties: local accuracy, missingness, and consistency. SHAP values unify six disparate existing methods (including LIME, DeepLIFT, and layer-wise relevance propagation) under a common mathematical structure, and the paper proposes new model-agnostic estimation methods (Kernel SHAP — connecting Shapley values to weighted linear regression) and model-specific approximations (Deep SHAP — composing SHAP values through a deep network via backpropagation) that improve computational efficiency and/or alignment with human intuition. User studies on Amazon Mechanical Turk demonstrate that SHAP values match human explanations more closely than LIME or DeepLIFT on simple logical models (fever-and-cough sickness scoring; max-function profit allocation), establishing that methods violating the consistency property produce attributions that deviate from human judgment in predictable ways.

## 2. Context and Motivation

### The Core Problem: Complex Models Are Accurate but Black Boxes

The paper grapples with a fundamental tension in applied machine learning that has intensified as models have grown more sophisticated: **the trade-off between accuracy and interpretability**. High-stakes applications increasingly demand both — models that match the predictive performance of modern ensembles and deep neural networks, while also providing explanations that let humans understand, trust, and act on individual predictions.

This is not merely an academic concern. The paper opens by stating that "understanding why a model makes a certain prediction can be as crucial as the prediction's accuracy in many applications" (Section 1). The authors list three concrete motivations for wanting interpretable predictions: engendering **appropriate user trust**, providing **insight into how a model may be improved**, and supporting **understanding of the process being modeled**. Each carries different practical weight depending on the domain:

- **User trust** matters when models inform decisions with personal consequences — medical diagnosis, loan approvals, criminal sentencing. A doctor who cannot understand *why* a model predicts high risk will reasonably hesitate to act on that prediction, regardless of the model's reported AUC.
- **Model improvement** matters during development. When a model fails on specific examples, an explanation of *which features drove the error* lets engineers diagnose whether the failure stems from data quality issues, feature engineering problems, or fundamental model limitations.
- **Process understanding** matters in scientific applications where the model is a tool for discovery — a geneticist using a predictor to identify disease-relevant genes needs feature importance scores that reflect genuine biological mechanisms, not artifacts of model architecture.

The paper describes the practical manifestation of this tension: "in some applications, simple models (e.g., linear models) are often preferred for their ease of interpretation, even if they may be less accurate than complex ones." This is a revealing sentence. It acknowledges that practitioners are *actively choosing worse predictive performance* because they cannot understand what complex models are doing. Each time this choice is made, it represents a concrete cost — misclassified patients, denied loans, or missed scientific discoveries — paid purely for the lack of interpretability tools.

### The Growing Urgency: Big Data Amplifies Both Sides of the Trade-Off

The paper locates itself at a specific historical moment in machine learning practice. "The growing availability of big data has increased the benefits of using complex models, so bringing to the forefront the trade-off between accuracy and interpretability of a model's output" (Section 1). This is a carefully framed claim: big data does not *create* the trade-off, but it *amplifies* it in both directions simultaneously.

On one side, bigger datasets make complex models more valuable. A deep neural network trained on millions of examples can find patterns that a linear model on the same data simply cannot represent. The accuracy gap widens, increasing the opportunity cost of choosing an interpretable but weaker model.

On the other side, bigger datasets often come from domains where interpretability matters most. The very applications that generate massive data — electronic health records, financial transactions, social media content moderation — are also applications where model decisions carry high stakes and where regulators, users, and developers demand explanations.

The paper thus positions itself at a pressure point: the forces pulling toward complex models (accuracy, data scale) and the forces pulling toward simple models (interpretability, trust) are both intensifying. A principled framework for interpreting complex models *post hoc* — after training, at prediction time — would release this pressure by decoupling the model's internal complexity from the explanation's comprehensibility.

### The Existing Landscape: Many Methods, No Unifying Understanding

By 2017, when this paper was published, the problem of interpreting complex model predictions had attracted substantial attention. The paper cites six distinct methods that had been recently proposed: LIME (Ribeiro et al., 2016), DeepLIFT (Shrikumar et al., 2016, 2017), Layer-Wise Relevance Propagation (Bach et al., 2015), Shapley regression values (Lipovetsky and Conklin, 2001), Shapley sampling values (Štrumbelj and Kononenko, 2014), and Quantitative Input Influence (Datta et al., 2016). Each arrived at feature attribution — assigning a number to each input feature indicating its contribution to a specific prediction — through different conceptual routes and with different estimation procedures.

The paper's central diagnosis is that despite the proliferation of methods, **the field lacked a framework for understanding how these methods relate and when one is preferable to another**. This is stated explicitly: "it is often unclear how these methods are related and when one method is preferable over another" (Section 1). The problem is not that individual methods are wrong — it is that the space of methods is fragmented, with each new approach introducing its own terminology, its own justification, and its own empirical validation procedure, making it impossible for practitioners to make informed choices or for researchers to build on previous work systematically.

This fragmentation carries several specific costs that the paper does not enumerate explicitly but that are implied by its analysis:

**1. Redundant development effort.** Methods developed independently in different subcommunities (Shapley-based game theory methods in one literature, gradient-based propagation methods in the deep learning literature, local approximation methods in the interpretable ML literature) were addressing the same underlying problem with overlapping solutions, but the overlaps were not recognized. Developers of a new method could not easily identify which aspects of their approach were genuinely novel and which merely rediscovered existing principles under different names.

**2. Hidden property violations.** Without a unified framework specifying the properties that a "good" explanation should satisfy, individual methods could violate important desiderata without their authors or users realizing it. The paper later demonstrates that both LIME and DeepLIFT, as originally proposed, compute values that violate consistency or local accuracy — but this would only be discoverable with a formal statement of what properties matter and a way to check whether a given method satisfies them.

**3. Benchmark incommensurability.** When methods are framed in different mathematical languages, it is difficult to design fair comparisons. Is a LIME explanation "better" than a DeepLIFT explanation? By what metric? Without a shared definition of what explanation quality means, papers could talk past each other, each reporting favorable results on whatever evaluation criteria their method was designed to optimize.

### Where Specific Prior Approaches Fall Short

The paper does not simply observe that the field is fragmented — it identifies specific limitations in the methods it unifies. The text provides enough detail about each to understand what the SHAP framework contributes beyond them.

#### LIME: Heuristic Choices That Violate Key Properties

LIME (Local Interpretable Model-agnostic Explanations), introduced by Ribeiro et al. (2016), explains a prediction by fitting a locally weighted linear model around the input of interest. The explanation model is additive (matching Definition 1 in the paper) and model-agnostic: it only requires the ability to query the original model on perturbed versions of the input, making it applicable to any classifier or regressor.

The paper acknowledges LIME's influence, particularly the concept of **local explanations** and the use of **simplified inputs** (binary vectors indicating feature presence/absence) mapped back to the original input space. However, the paper identifies a specific weakness: LIME's choices for the loss function $L$, the weighting kernel $\pi_{x'}$, and the regularization term $\Omega$ in its optimization objective (Equation 2 in the paper) are **made heuristically**. The objective is:

$$\xi = \arg\min_{g \in G} L(f, g, \pi_{x'}) + \Omega(g)$$

LIME picks these components to work well in practice but without a theoretical guarantee that the resulting attributions satisfy any particular set of desirable properties. One consequence the paper demonstrates empirically (Section 5, Figure 3) is that "values from LIME can differ significantly from SHAP values that satisfy local accuracy and consistency." The implication is that LIME's heuristic choices produce attributions that can violate the consistency property — meaning that if a feature's contribution increases or stays the same regardless of other inputs (Property 3), LIME's attribution for that feature might paradoxically *decrease*. This is not merely a theoretical subtlety; the paper's user study (Figure 4) shows that such violations lead to explanations that diverge from human judgments.

#### DeepLIFT: Heuristic Linearization Without a Unifying Principle

DeepLIFT (Deep Learning Important FeaTures), introduced by Shrikumar et al. (2016, 2017), addresses a different limitation: LIME's model-agnostic approach, while flexible, is computationally expensive for deep networks because it requires many forward passes through the model on perturbed inputs. DeepLIFT instead exploits the **compositional structure** of neural networks, propagating attribution scores backward through the network's layers using a set of rules that define how each component (linear layers, activation functions, max pooling) contributes to the output change relative to a reference input.

The paper identifies two aspects of DeepLIFT that the SHAP framework clarifies:

**First**, DeepLIFT's "summation-to-delta" property — that the sum of all feature attributions equals the difference between the model's output on the actual input and its output on a reference input — makes it an additive feature attribution method (matching Definition 1). This was not previously recognized as connecting DeepLIFT to the broader family of additive explanation methods.

**Second**, and more critically, DeepLIFT's back-propagation rules for linearizing non-linear components were **heuristically chosen**. The authors state this explicitly in Section 4.2: "Its back-propagation rules defining how each component is linearized are intuitive but were heuristically chosen." The max pooling function provides a concrete example: DeepLIFT needed a rule for how to attribute the output of a max operation to its inputs, and the original rule was designed to be reasonable but lacked a formal justification. The paper's user study (Figure 4B) demonstrates that this heuristic linearization produces attributions that differ from human judgments on a simple max-function problem, while SHAP-derived attributions match human intuition closely.

The paper's treatment of DeepLIFT is particularly instructive because it shows how the unified framework enables *improvement* of existing methods. Rather than rejecting DeepLIFT, the paper shows that by replacing heuristic linearization rules with SHAP-value-based linearization (Deep SHAP), the attributions better match human judgment (Figure 5, comparing "Orig. DeepLIFT" with "New DeepLIFT").

#### Layer-Wise Relevance Propagation: A Special Case Without the General Principle

Layer-Wise Relevance Propagation (LRP), introduced by Bach et al. (2015), propagates relevance scores from the output layer backward to the input, using rules that conserve relevance across layers. The paper notes that as observed by Shrikumar et al., LRP is "equivalent to DeepLIFT with the reference activations of all neurons fixed to zero." This makes LRP another additive feature attribution method, but one that inherits both the strengths (computational efficiency for deep networks) and weaknesses (heuristic rules, no satisfaction of consistency) of DeepLIFT, with the additional limitation of being restricted to a specific reference value (zero).

#### Shapley-Based Methods: Correct in Principle, Computationally Demanding

Three methods — Shapley regression values (Lipovetsky and Conklin, 2001), Shapley sampling values (Štrumbelj and Kononenko, 2014), and Quantitative Input Influence (Datta et al., 2016) — draw directly on Shapley values from cooperative game theory. They compute feature attributions by considering all possible subsets of features and measuring how each feature changes the model's prediction when added to different subsets.

The paper identifies these methods as sharing the same underlying explanation model as LIME and DeepLIFT (Definition 1), which means they are all members of the same additive feature attribution class. This is a non-obvious unification: game-theoretic methods developed for linear models and applied to general models via sampling, and gradient-based propagation methods developed for deep networks, are targeting the same explanation model.

However, the Shapley-based methods have a practical limitation: **exponential computational cost**. Computing exact Shapley values requires evaluating the model on all $2^{|F|}$ subsets of features (Equation 4 in the paper). Shapley sampling values and Quantitative Input Influence reduce this by sampling subsets and estimating the effect of removing features, but the paper shows in Section 5 (Figure 3) that these sampling-based estimates require many model evaluations to converge. The computational burden limits applicability to models with many features or where model evaluations are expensive (e.g., large deep networks).

Additionally, these methods require defining what it means for a feature to be "absent" from a model — a concept that is well-defined for linear models (the coefficient is excluded from the fit) but ambiguous for arbitrary black-box models. The Shapley sampling approach handles this by marginalizing over training data: the effect of removing feature $i$ is approximated by averaging the model's predictions when feature $i$ takes values drawn from its marginal distribution rather than its observed value. This is a reasonable practical approximation but introduces assumptions (feature independence) that the paper later formalizes.

### The Unrecognized Unity: A Deeper Problem Than It Appears

The paper's key motivating insight is that the fragmentation described above is not merely cosmetic — it masks a **structural unity** that has theoretical consequences. All six methods produce explanations that fit Definition 1: a linear function of binary variables where each coefficient $\phi_i$ represents the effect attributed to feature $i$. This is the "previously unappreciated unity" the paper announces (Section 2).

The significance of this unity is that it lets the paper apply results from cooperative game theory — specifically, the uniqueness of Shapley values — to the *entire class* of methods. Young (1985) proved that Shapley values are the unique set of values satisfying three axioms in cooperative games. The paper adapts this proof to show that Shapley values are the unique additive feature attribution method satisfying local accuracy (the explanation matches the original model's output), missingness (absent features get zero attribution), and consistency (if a model changes so a feature's contribution never decreases, its attribution should not decrease).

This means that **methods in this class that are not based on Shapley values necessarily violate at least one of these properties**. The paper does not claim that LIME or DeepLIFT are "wrong" in any absolute sense — it claims that to the extent one believes local accuracy and consistency are desirable properties for explanations, Shapley values are the *only* way to satisfy them within the additive attribution framework. This transforms a messy landscape of competing methods into a clean theoretical picture: there is a space of possible additive attribution methods, and exactly one point in that space satisfies the three properties. Methods can be evaluated by how close they come to that point and by *which* properties they sacrifice when they deviate.

### How the Paper Positions Itself

The paper's positioning is unusual and ambitious: rather than proposing a new method to compete with existing ones, it proposes a **framework that subsumes them**. The introduction explicitly frames this as a contribution that "brings clarity to the growing space of methods" (Section 1). The paper's three main results reflect this:

1. **Unification**: Showing that six disparate methods share the same explanation model form (Definition 1) — a taxonomic contribution that makes visible relationships that were previously obscured by different terminology and motivation.

2. **Theoretical grounding**: Proving that within this class, Shapley values are the unique solution satisfying local accuracy, missingness, and consistency (Theorem 1) — a normative contribution that identifies *which* of the possible additive attributions are maximally well-behaved.

3. **Practical improvement**: Using insights from the unification to develop better estimation methods (Kernel SHAP for model-agnostic regression-based estimation; Deep SHAP for compositional approximation through deep networks) and demonstrating that these produce explanations more aligned with human intuition than prior methods (Sections 4-5).

The paper's stance toward prior work is notably constructive. It does not dismiss LIME or DeepLIFT as wrong; instead, it shows that they are *approximations* of SHAP values — in some cases, approximations that diverge enough to violate important properties. The contributions are both theoretical (explaining *why* methods should target SHAP values) and practical (showing *how* to estimate them efficiently). This dual contribution — a normative theory of what explanations should look like, plus tractable algorithms for computing them — is what distinguishes the SHAP framework from purely theoretical work on Shapley values in model interpretation.

A final subtlety in the paper's positioning: SHAP values are presented as the Shapley values of a **conditional expectation function** of the original model (Section 4). That is, $f_x(z') = \mathbb{E}[f(z) \mid z_S]$, where $S$ is the set of features indicated as present in the simplified input $z'$. This is a specific — and practically motivated — choice. By defining the value of a feature subset as the expected model output conditioned on the known feature values (marginalizing over unknown features), SHAP directly connects to how Shapley regression and Shapley sampling operate. The paper acknowledges that this requires approximating conditional expectations (Equation 9-12), and it makes explicit which assumptions (feature independence, model linearity) different approximation methods adopt. This explicit handling of what "missing feature" means addresses the ambiguity that earlier Shapley-based methods grappled with implicitly.

## 3. Technical Approach

### 3.1 Reader Orientation

This is primarily a **theoretical unification and algorithmic improvement paper** whose core idea is that all model-agnostic feature attribution methods that express explanations as a sum of per-feature contributions (an "additive feature attribution model") share a single mathematical structure, and within that structure, Shapley values from cooperative game theory provide the unique set of attributions satisfying three natural axioms: local accuracy, missingness, and consistency. The paper shows that existing methods like LIME and DeepLIFT are approximations of these Shapley values — in some cases, approximations that unintentionally violate the axioms — and then develops new estimation algorithms (Kernel SHAP, Deep SHAP) that either recover the Shapley values exactly or produce better approximations than prior approaches.

### 3.2 Big-Picture Architecture (Diagram in Words)

The SHAP framework consists of five interconnected components:

1. **The Original Prediction Model ($f$)** — the black-box model whose predictions need explaining (e.g., a deep neural network, a tree ensemble, or any arbitrary function). It takes an input $x$ and produces an output $f(x)$, which could be a regression value or a class probability.

2. **The Simplified Input Mapping ($h_x$)** — a function that converts a binary vector $z' \in \{0, 1\}^M$ (where $M$ is the number of features, and each bit indicates whether a feature is "present" or "absent") into the original input space that $f$ can consume. This mapping is specific to the current input $x$ being explained and encodes what it means for a feature to be "missing" (e.g., for text, replacing a word with zero count; for images, replacing a superpixel with a background average).

3. **The Explanation Model ($g$)** — a linear function of the binary simplified inputs, $g(z') = \phi_0 + \sum_{i=1}^M \phi_i z_i'$, whose coefficients $\phi_i$ are the feature attribution values. This is the output the user sees: each $\phi_i$ represents the contribution of feature $i$ to pushing the prediction away from the baseline $\phi_0$.

4. **The Conditional Expectation Function ($f_x$)** — the mechanism for evaluating the original model on subsets of features. When a simplified input $z'$ has features in set $S$ "present" and others "absent," SHAP defines $f_x(z') = \mathbb{E}[f(z) \mid z_S]$, the expected model output conditioned on the known feature values. This is what turns the Shapley value formula into a concrete computation.

5. **The Estimation Algorithm** — the procedure that actually computes the $\phi_i$ values given $f$, $h_x$, and $x$. The paper provides several: Kernel SHAP (model-agnostic, uses weighted linear regression), Linear SHAP (closed-form for linear models), Deep SHAP (compositional backpropagation for deep networks), and Max SHAP (efficient exact computation for max functions).

Information flows as follows: the user provides an input $x$ and a model $f$ → the simplified input mapping $h_x$ defines what "presence/absence" means for each feature → the conditional expectation function $f_x$ evaluates $f$ on all feature subsets (or a sample thereof) → the estimation algorithm solves for the $\phi_i$ that satisfy the Shapley axioms → the explanation model $g$ presents these $\phi_i$ as the feature attributions.

### 3.3 Roadmap for the Deep Dive

- **First**, the paper's central formal move: viewing all explanations as **explanation models** (Definition 1) and defining the class of additive feature attribution methods. This is the conceptual scaffold — everything else hangs on understanding that LIME, DeepLIFT, and Shapley-based methods all fit the same equation $g(z') = \phi_0 + \sum \phi_i z_i'$. We need to understand what $z'$ and $h_x$ are concretely, because the paper's theoretical results apply to *any* method using this form.

- **Second**, the **three axioms** (local accuracy, missingness, consistency) and Theorem 1, which proves Shapley values are the unique solution satisfying all three. This is the normative core: if you believe these properties are desirable, Shapley values are the only game in town. We need to understand each axiom's operational meaning and why violating it produces unintuitive explanations.

- **Third**, the **Shapley value formula** itself (Equation 8) — what it computes, why it averages over all feature subsets with those particular combinatorial weights, and how conditional expectations ($\mathbb{E}[f(z) \mid z_S]$) make it concrete for black-box models. This introduces the key practical challenge: computing $2^M$ expectations is infeasible.

- **Fourth**, the **Kernel SHAP** method, which is the paper's most important algorithmic contribution: proving that Shapley values can be recovered by solving a specific weighted linear regression problem (Theorem 2). This connects game theory to regression, enables sample-efficient estimation, and reveals exactly how LIME's heuristic kernel differs from the Shapley kernel.

- **Fifth**, the **model-specific approximations**: Linear SHAP (closed-form for linear models), Deep SHAP (composing SHAP values through a deep network using a Shapley-value-based chain rule), and Max SHAP (an $O(M^2)$ algorithm for the max function). These show how the framework enables computational shortcuts by exploiting model structure.

### 3.4 Detailed, Sentence-Based Technical Breakdown

#### The Explanation Model Perspective (Definition 1 and the Simplified Input Mapping)

The paper's foundational move is to reframe the problem of interpreting a model prediction not as "computing feature importance scores" but as **building an explanation model** — a separate, simpler model $g$ that approximates the original model $f$ in the vicinity of the input being explained. This reframing matters because it makes explicit that every interpretation method *is itself a model*, with its own functional form, its own input representation, and its own fitting procedure. Once you see this, you can ask: what class of models do current methods use, and what properties should that class have?

**Definition 1 (Additive Feature Attribution Methods).** The paper defines the class it will study:

$$g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z_i'$$

where $z' \in \{0, 1\}^M$ is a binary vector of simplified input features, $M$ is the number of such features, and $\phi_i \in \mathbb{R}$ are the attributed effects.

**What this means operationally.** The explanation model $g$ is a linear function of binary variables. Each $z_i'$ is either 0 or 1, indicating whether simplified feature $i$ is "absent" or "present" in the explanation. The coefficient $\phi_i$ is the amount that feature $i$ contributes to shifting the output away from the baseline $\phi_0$ (which is the model's output when all simplified features are absent, i.e., $z' = \mathbf{0}$). The prediction $f(x)$ is approximated by summing these per-feature contributions.

**Why this form.** The paper does not claim this is the only possible explanation model class — it observes that this is the class that **six existing methods already use**, whether their authors realized it or not. The binary input representation $z'$ is motivated by a natural notion of feature presence/absence: for any given prediction, we want to know how each feature's *being present* (as opposed to being missing/replaced by a reference value) changed the output. The additivity constraint means there are no interaction terms — each feature's effect is independent — which makes the explanation comprehensible to humans. A human can look at the list of $\phi_i$ values and mentally sum them to see how the prediction was built up.

**The simplified input mapping $h_x$.** Critically, the binary variables $z'$ do not directly correspond to the original model's input features. There is an intermediate mapping $x = h_x(z')$ that converts a simplified input $z'$ (a pattern of feature presence/absence) into a concrete input in the original model's input space. This mapping is subscripted by $x$ because it is **specific to the current input being explained**. The paper states: "Local methods try to ensure $g(z') \approx f(h_x(z'))$ whenever $z' \approx x'$," and notes that $h_x(x') = x$ by construction (when all simplified features are present, the mapping recovers the original input).

Different methods in the literature use different $h_x$ mappings, and understanding these differences is essential to seeing how they all fit Definition 1:

- **LIME for text (bag of words).** $h_x$ converts a binary vector (word present/absent) into the original word count representation: if $z_i' = 1$, the word count for feature $i$ is set to its value in the original input $x$; if $z_i' = 0$, that word count is set to zero (representing word removal). This is a natural notion of "absence" for text: removing a word means setting its count to zero.

- **LIME for images (superpixels).** $h_x$ treats the image as a set of superpixels (contiguous pixel regions). If $z_i' = 1$, superpixel $i$ keeps its original pixel values; if $z_i' = 0$, superpixel $i$ is replaced with an "average of neighboring pixels," which LIME treats as representing "missingness." This is more heuristic than the text case — there is no uniquely natural definition of an absent image region — but the mapping still fits the binary-to-original-input form.

- **DeepLIFT.** $h_x$ maps $z_i' = 1$ to the original feature value $x_i$, and $z_i' = 0$ to a user-chosen reference value $r_i$ (a "typical uninformative background value"). The reference value is manually specified and represents what the feature would be if it provided no information. The explanation then explains the difference $\Delta o = f(x) - f(r)$ rather than the absolute output $f(x)$.

- **Layer-Wise Relevance Propagation.** Equivalent to DeepLIFT with all reference values fixed to zero: $h_x$ maps $z_i' = 1$ to $x_i$ and $z_i' = 0$ to $0$.

- **Shapley regression values.** $h_x$ maps $z_i' = 1$ to "include feature $i$ in the model" and $z_i' = 0$ to "exclude feature $i$ from the model." For linear regression, this means the feature is either included as a predictor or excluded from the fit. The $\phi_0 = f_\emptyset(\emptyset)$ term is the model's prediction when trained with no features (i.e., just the intercept).

- **Shapley sampling values and Quantitative Input Influence.** These handle black-box models by approximating "feature removal" as marginalizing over the training data distribution: when $z_i' = 0$, instead of literally removing the feature, they replace $x_i$ with values drawn randomly from the training set marginal distribution for that feature. This integrates over possible values the feature could have taken, approximating what the model would predict if it did not know the feature's actual value.

**The crucial distinction: $x'$ vs. $z'$.** The paper uses $x'$ to denote the simplified input corresponding to the *original* input $x$ — that is, $x'$ is the binary vector with all features present ($x_i' = 1$ for all $i$, indicating that every feature is at its original value). Meanwhile, $z'$ ranges over arbitrary simplified inputs (any pattern of present/absent features). The mapping $h_x$ must satisfy $h_x(x') = x$, meaning that when all features are "on," the original input is recovered exactly.

**Why this perspective is powerful.** By casting every method's output as an instance of the same linear-binary model form, the paper can now ask: among all possible choices of $\phi_i$ values for this model, which ones are "correct"? This is where the axioms enter.

---

#### Three Axioms That Pin Down the Unique Solution (Properties 1–3 and Theorem 1)

The paper's theoretical contribution is proving that **Shapley values are the unique set of $\phi_i$ satisfying three natural properties**. This is adapted from Young (1985)'s characterization of Shapley values in cooperative game theory, with one additional property (missingness) needed to make the proof work in the additive feature attribution setting.

**Property 1: Local Accuracy.** The explanation must exactly match the original model's output when evaluated on the simplified input corresponding to the full original input:

$$f(x) = g(x') = \phi_0 + \sum_{i=1}^{M} \phi_i x_i'$$

Since $x_i' = 1$ for all features in the original input (all features are "present"), this simplifies to:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

**What this means operationally.** When you add up the baseline value $\phi_0$ (the model's output with no features) plus all the per-feature attributions $\phi_i$, the sum must exactly equal the model's prediction $f(x)$. The explanation accounts for every unit of the prediction; nothing is left unexplained and nothing is double-counted.

**Why this matters.** If local accuracy is violated, the sum of attributions does not equal the prediction, which means the explanation is internally inconsistent: a user looking at the attribution scores cannot mentally add them to recover what the model actually predicted. LIME's heuristic loss function and regularization (Equation 2 with heuristically chosen $\pi_{x'}$ and $\Omega$) do not enforce this exactly — LIME trades off fidelity to $f$ against model complexity, so its attributions will generally not sum to $f(x)$. The paper later shows this leads to explanations that disagree with human judgments.

The paper defines $\phi_0 = f(h_x(\mathbf{0}))$, which is the model's output when all simplified inputs are "off" — i.e., the baseline prediction with no features present. For DeepLIFT, this is $f(r)$, the model's output on the reference input. For Shapley regression, this is $f_\emptyset(\emptyset)$, the intercept-only model's prediction.

**Property 2: Missingness.** If a simplified feature is absent, its attribution must be zero:

$$x_i' = 0 \implies \phi_i = 0$$

**What this means operationally.** Features that are already "missing" in the simplified input cannot have any attributed effect. Only features that are present can contribute to shifting the prediction away from the baseline. This is a bookkeeping property: it prevents the explanation from attributing effect to features it claims are not there.

**Why this matters.** The paper notes that all methods discussed in Section 2 already satisfy missingness, so this property is less distinctive than the others. However, it is logically required for the uniqueness proof: without it, you could have explanations where absent features receive non-zero attributions, which would be nonsensical (how can a removed word affect the prediction?) but might still satisfy local accuracy and consistency if the attributions for absent features were accounted for elsewhere. Missingness rules out such degenerate solutions.

**Property 3: Consistency.** This is the most subtle and consequential property. It states that if the model changes so that a particular simplified feature's marginal contribution increases or stays the same *regardless of which other features are present*, then that feature's attribution should not decrease.

Formally: let $f_x(z') = f(h_x(z'))$ be the model evaluated on the simplified input (this redefinition is purely for notational convenience in stating the property). For any two models $f$ and $f'$, if:

$$f'_x(z') - f'_x(z' \setminus i) \geq f_x(z') - f_x(z' \setminus i)$$

for all $z' \in \{0, 1\}^M$ (where $z' \setminus i$ means setting $z_i' = 0$), then $\phi_i(f', x) \geq \phi_i(f, x)$.

**What this means in plain language.** For every possible subset of other features being present or absent, the marginal contribution of feature $i$ — defined as the difference between the model's output with feature $i$ present and with it absent, holding all other features fixed — is at least as large in model $f'$ as it is in model $f$. If this holds universally (for all $2^{M-1}$ possible subsets of other features), then feature $i$'s total attribution in $f'$ should be at least as large as its attribution in $f$. The attribution cannot go down when the feature becomes uniformly more important.

**Why this matters — a concrete violation scenario.** Suppose we have a model that predicts sickness severity from symptoms. In the original model $f$, having both fever and cough yields a score of 2, having either alone yields 5, and having neither yields 0. Now consider a modified model $f'$ that is identical except that having both fever and cough now yields a score of 10 instead of 2. Feature "fever" now has a larger or equal marginal contribution in every subset: when cough is absent, fever's contribution is 5 in both models; when cough is present, fever's contribution is 3 in $f$ (5 → 2? No, let's trace this carefully) — actually, when cough is present, adding fever in $f$ changes the prediction by (score with both) minus (score with cough only), and in $f'$ it changes by (10 minus 5). If the new score is higher, the marginal contribution increased. Consistency requires that fever's attribution in $f'$ should be at least as large as in $f$.

The paper demonstrates empirically (Section 5, Figure 4) that both LIME and original DeepLIFT violate consistency in ways that produce attributions disagreeing with human judgment. For instance, in the max-function problem (Figure 4B), DeepLIFT's heuristic linearization of the max function produces attributions that do not satisfy consistency, leading to credit assignments that humans find unintuitive.

**Theorem 1 (Uniqueness).** The paper states:

> Only one possible explanation model $g$ follows Definition 1 and satisfies Properties 1, 2, and 3:

$$\phi_i(f, x) = \sum_{z' \subseteq x'} \frac{|z'|! (M - |z'| - 1)!}{M!} \left[ f_x(z') - f_x(z' \setminus i) \right]$$

where $|z'|$ is the number of non-zero entries in $z'$, and $z' \subseteq x'$ represents all $z'$ vectors where the non-zero entries are a subset of the non-zero entries in $x'$.

**What this equation computes.** This is exactly the classic Shapley value formula from cooperative game theory. The term $f_x(z') - f_x(z' \setminus i)$ is the **marginal contribution** of feature $i$ when added to the set of features that are "on" in $z'$ (but note: $z' \setminus i$ means setting $z_i' = 0$, so $z'$ must have had $z_i' = 1$ for this difference to be non-trivial — the sum is over $z'$ where feature $i$ is present). The weight $\frac{|z'|!(M-|z'|-1)!}{M!}$ is a combinatorial factor that averages over all possible orders in which features could be added.

**Why these weights.** The weight has a specific combinatorial interpretation: consider all $M!$ possible orderings (permutations) of the $M$ features. For a given ordering, imagine adding features one at a time in that order, starting from the empty set, and measuring how much each newly added feature changes the prediction. The Shapley value for feature $i$ is the *average* of these marginal contributions across all orderings. For a fixed set size $|z'|$, the marginal contribution $f_x(z') - f_x(z' \setminus i)$ (where $z'$ has feature $i$ present and $|z'|-1$ other features present) appears in all orderings where feature $i$ is preceded by exactly those $|z'|-1$ other features and followed by the remaining $M - |z'|$ features. The number of such orderings is $(|z'|-1)! \times (M - |z'|)!$. Summing over all subsets containing $i$ and dividing by $M!$ gives the formula. This averaging over orderings is what ensures consistency: if a feature's marginal contribution increases in every subset, its average across all orderings must also increase.

**The proof structure.** Theorem 1 follows from Young (1985), who proved that Shapley values are the unique values satisfying three axioms in cooperative games: efficiency (total value is distributed among players), symmetry (identical contributions get identical values), and monotonicity (stronger contributions get larger values). The paper maps these game-theoretic axioms to the three properties: efficiency → local accuracy, dummy (zero-contribution players get zero) → missingness, and monotonicity → consistency. The paper notes that one of Young's axioms (symmetry) is actually redundant given the other axioms in this setting (see Supplementary Material of the original paper), leaving three properties that are both necessary and sufficient for uniqueness.

---

#### The SHAP Value Definition and the Conditional Expectation Function

With the uniqueness result established, the paper defines **SHAP values** as a specific instantiation of Equation 8, where the function $f_x(z')$ is defined using **conditional expectations**:

$$f_x(z') = f(h_x(z')) = \mathbb{E}[f(z) \mid z_S]$$

where $S$ is the set of non-zero indices in $z'$ (the features that are "present"), and $z_S$ denotes the original input with the known feature values fixed and the unknown features treated as random variables to be marginalized over.

**What this means concretely.** When evaluating the model on a subset of features $S$, we cannot simply "remove" the features not in $S$ — most models require complete input vectors. Instead, we compute the **expected value** of the model's output, where the expectation is taken over the distribution of the unknown features conditioned on the known feature values. For example, if we are explaining a prediction for a patient with age=65, blood_pressure=140, and cholesterol=200, and we want to know what the model would predict knowing only age=65 (i.e., $S = \{\text{age}\}$), we average the model's predictions over the joint distribution of blood pressure and cholesterol among patients who are 65 years old.

**Why conditional expectations and not marginal expectations.** An alternative — used by Shapley sampling values and Quantitative Input Influence — is to use $\mathbb{E}_{z_{\bar{S}}}[f(z)]$, which assumes feature independence. This marginalizes over the unknown features using their *marginal* distribution, ignoring correlations with the known features. For the age-65 example, this would average over all blood pressure and cholesterol values in the entire population, regardless of age. The conditional expectation approach is more faithful to the data-generating process: when we know age=65, we should use the distribution of other features conditional on that knowledge, not the unconditional distribution.

The paper makes this choice explicit and frames subsequent approximations as simplifying assumptions on top of this definition:

> "Implicit in this definition of SHAP values is a simplified input mapping, $h_x(z') = z_S$, where $z_S$ has missing values for features not in the set $S$. Since most models cannot handle arbitrary patterns of missing input values, we approximate $f(z_S)$ with $\mathbb{E}[f(z) \mid z_S]$."

**The approximation chain (Equations 9–12).** The paper spells out a progression of increasingly strong assumptions for computing SHAP values in practice:

**Exact definition (Equation 9–10):**
$$f(h_x(z')) = \mathbb{E}[f(z) \mid z_S] = \mathbb{E}_{z_{\bar{S}} \mid z_S}[f(z)]$$

where $\bar{S}$ is the set of features not in $S$. This is the true conditional expectation — the expected model output given the known feature values.

**Feature independence assumption (Equation 11):**
$$\mathbb{E}_{z_{\bar{S}} \mid z_S}[f(z)] \approx \mathbb{E}_{z_{\bar{S}}}[f(z)]$$

This assumes that the features not in $S$ are independent of the features in $S$, so conditioning on $z_S$ does not change their distribution. The expectation simplifies to integrating over the marginal distribution of the unknown features. This is the assumption used by Shapley sampling values (Štrumbelj and Kononenko, 2014), LIME (Ribeiro et al., 2016), DeepLIFT (Shrikumar et al., 2017), and Quantitative Input Influence (Datta et al., 2016). It is a computational convenience: you can estimate the marginal distribution from training data and sample from it without modeling feature dependencies.

**Model linearity assumption (Equation 12):**
$$\mathbb{E}_{z_{\bar{S}}}[f(z)] \approx f([z_S, \mathbb{E}[z_{\bar{S}}]])$$

If we additionally assume that $f$ is approximately linear (or we are willing to linearize it), then the expectation of $f$ over the unknown features equals $f$ evaluated at the expected values of those features. This means we can simply fill in the mean value (or reference value) for unknown features and make a single forward pass through the model, rather than sampling many values and averaging. LIME's simplified input mapping — where "absent" features are replaced by a reference value (zero for text, average neighbor pixels for images) — corresponds to this linearity assumption.

**The design choice: which approximation level to use.** The paper provides methods at different points on this approximation chain. Kernel SHAP (Section 4.1) can work with any of these approximations, depending on how the practitioner implements $f_x(z')$. Deep SHAP (Section 4.2) uses the feature independence + model linearity approximation (Equation 11 + 12) to achieve computational efficiency for deep networks. The paper does not claim one approximation level is universally correct — it makes the assumptions explicit so users can decide based on their computational budget and their tolerance for approximation error.

---

#### Kernel SHAP: Recovering Shapley Values via Weighted Linear Regression

The paper's most novel algorithmic contribution is **Kernel SHAP**, which proves that Shapley values can be computed by solving a specific weighted linear regression problem. This connects two previously separate literatures — game-theoretic Shapley values and regression-based model explanation (LIME) — and provides a more sample-efficient estimation method than classical Shapley sampling.

**The starting point: LIME's optimization objective.** LIME fits its explanation model by minimizing:

$$\xi = \arg\min_{g \in G} L(f, g, \pi_{x'}) + \Omega(g)$$

where $L$ is a loss function measuring how well $g$ approximates $f$ on a set of samples $z'$ in the simplified input space, $\pi_{x'}$ is a weighting kernel that gives higher weight to samples $z'$ that are "close" to the original simplified input $x'$, and $\Omega$ is a regularization term penalizing model complexity. For linear LIME, $g$ takes the additive form of Equation 1, and $L$ is a squared loss:

$$L(f, g, \pi_{x'}) = \sum_{z' \in Z} [f(h_x(z')) - g(z')]^2 \pi_{x'}(z')$$

where $Z$ is a set of sampled simplified inputs. LIME chooses $\pi_{x'}$ heuristically (e.g., an exponential kernel based on cosine distance in the simplified input space) and uses ridge regression or lasso for $\Omega$.

**The key question the paper asks.** Since we already know from Theorem 1 that Shapley values are the unique additive attribution satisfying Properties 1–3, and LIME's explanation model has the same additive form, under what choices of $L$, $\pi_{x'}$, and $\Omega$ will LIME's regression *recover* the Shapley values?

The answer — and this is the paper's major insight — is that there exists a **specific weighting kernel** (different from LIME's heuristic kernel) and a **specific choice of no regularization** ($\Omega = 0$) that makes the least-squares solution of Equation 2 exactly equal to the Shapley values.

**Theorem 2 (Shapley kernel).** The specific forms that make the solution of Equation 2 consistent with Properties 1 through 3 are:

$$\Omega(g) = 0$$

$$\pi_{x'}(z') = \frac{M - 1}{\binom{M}{|z'|} |z'| (M - |z'|)}$$

$$L(f, g, \pi_{x'}) = \sum_{z' \in Z} [f(h_x(z')) - g(z')]^2 \pi_{x'}(z')$$

where $|z'|$ is the number of non-zero elements in $z'$ (the number of features present in the simplified input).

**What this kernel does.** The Shapley kernel $\pi_{x'}(z')$ assigns a weight to each simplified input $z'$ based solely on the number of features present, $|z'|$. The weight is a symmetric function of $|z'|$ — all simplified inputs with the same number of "on" features get the same weight, regardless of *which* features are on.

**Why this particular functional form.** The derivation (in the paper's Supplementary Material) comes from solving for the weights that make the regression coefficients equal the Shapley formula (Equation 8). Intuitively, the weights must satisfy: (1) they go to infinity at the extremes $|z'| = 0$ and $|z'| = M$, which forces the regression to exactly fit the all-absent and all-present points (enforcing local accuracy and the $\phi_0$ definition); (2) they assign higher weight to small and large subsets than to medium-sized subsets, reflecting the combinatorial structure of the Shapley averaging weights.

**The infinity problem at the boundaries.** The paper notes that $\pi_{x'}(z') = \infty$ when $|z'| \in \{0, M\}$. This is because the binomial coefficient $\binom{M}{|z'|}$ appears in the denominator, and dividing by zero (or near-zero combinatorial terms) produces infinite weight. In practice, this is handled by **analytically eliminating two variables**: the constraints $\phi_0 = f_x(\emptyset)$ and $\sum_{i=0}^M \phi_i = f(x)$ are enforced explicitly, and only the remaining $M-1$ coefficients are solved via regression on the $z'$ vectors with $0 < |z'| < M$. This avoids having to handle infinite weights in the optimization.

**Why this is better than Shapley sampling.** The regression formulation estimates all $\phi_i$ jointly, which is more sample-efficient than classical Shapley sampling, which estimates each $\phi_i$ independently using separate sampling procedures. The joint estimation shares information across features: the regression must find a single set of coefficients that simultaneously explains the model's output across all sampled feature subsets, which acts as a form of implicit regularization. The paper's experiments (Section 5, Figure 3) confirm this empirically: Kernel SHAP converges to the true Shapley values with fewer model evaluations than Shapley sampling, particularly when $f$ depends on only a few features (sparse case).

**Regularization in practice.** The paper notes that while Theorem 2 specifies $\Omega(g) = 0$ (no regularization), in practice "regularization is added to the linear model" when using Kernel SHAP with limited samples (Figure 3 caption: "Kernel SHAP (using a debiased lasso)"). This is a practical compromise: with a finite sample of $z'$ vectors rather than all $2^M$, regularization helps prevent overfitting to the sampled subsets. The debiased lasso procedure fits a lasso-regression model (L1-regularized) to select features and then re-fits without regularization on the selected features to remove the L1 shrinkage bias. This retains the sparsity-inducing benefits of lasso while producing approximately unbiased coefficient estimates.

**How the Shapley kernel differs from LIME's heuristic kernel (Figure 2A).** The paper provides a visual comparison in Figure 2A, showing the Shapley kernel weight as a function of $|z'|$ (the number of features present) for $M = 15$ features. The kernel is symmetric: it assigns high weight to simplified inputs with very few features present (small $|z'|$) and to those with many features present (large $|z'|$), with lower weight in the middle. This is distinctly different from LIME's heuristically chosen kernel, which typically decays with distance from $x'$ (i.e., gives highest weight to $z'$ vectors close to the all-present vector). The Shapley kernel's U-shape reflects the combinatorial structure of Shapley values: marginal contributions when few other features are present and when almost all features are present carry more weight in the Shapley average than contributions in intermediate-size coalitions. LIME's distance-based kernel misses this structure entirely, which is why LIME coefficients diverge from Shapley values.

---

#### Model-Specific SHAP Approximations

While Kernel SHAP is model-agnostic (requiring only the ability to evaluate $f_x(z')$ for any $z'$), the paper also develops faster approximations that exploit knowledge of the model's internal structure. These are motivated by the observation that for certain model types, SHAP values can be computed analytically or nearly so, bypassing the need to sample and evaluate many feature subsets.

##### Linear SHAP

For linear models, SHAP values have a closed-form solution under the feature independence assumption (Equation 11). This is stated as Corollary 1:

> Given a linear model $f(x) = \sum_{j=1}^M w_j x_j + b$: $\phi_0(f, x) = b$ and $\phi_i(f, x) = w_j(x_j - \mathbb{E}[x_j])$.

**What this computes.** The baseline $\phi_0$ is simply the model's bias term $b$ (the prediction when all features are at their expected values, since the linear model's expectation is the bias plus weighted sum of expected feature values, and those cancel). Each feature's SHAP value is its weight coefficient multiplied by the difference between the feature's actual value and its expected (mean) value. If $x_j$ equals the mean $\mathbb{E}[x_j]$, the feature contributes nothing — it is "average" and does not shift the prediction. If $x_j$ is above its mean, the contribution is positive (for $w_j > 0$) or negative (for $w_j < 0$).

**Why this is the correct Shapley value for a linear model.** Under feature independence (Equation 11), the expected value of the model conditioned on a feature set $S$ is:

$$\mathbb{E}[f(z) \mid z_S] = b + \sum_{j \in S} w_j x_j + \sum_{j \notin S} w_j \mathbb{E}[x_j]$$

The marginal contribution of adding feature $i$ to set $S$ is always $w_i (x_i - \mathbb{E}[x_i])$, *regardless of which other features are already in* $S$. Since the marginal contribution is constant across all subsets, the Shapley value (which averages marginal contributions over subsets) is exactly this constant value. The independence assumption makes the marginal contribution subset-invariant for linear models.

**The connection to standard linear regression interpretation.** This result formalizes a common intuition: in a linear model, a feature's importance is its coefficient times its deviation from the mean. SHAP values recover this as the unique attribution satisfying the three axioms, which validates that the axioms produce intuitive results in the simple case where intuition is clear.

##### Low-Order SHAP

The paper briefly mentions that "linear regression using Theorem 2 has complexity $O(2^M + M^3)$, it is efficient for small values of $M$ if we choose an approximation of the conditional expectations." The $O(2^M)$ term comes from the need to enumerate all possible feature subsets (or at least a sample thereof), and the $O(M^3)$ term comes from solving the linear regression (matrix inversion). For small $M$ (say, $M \leq 20$), this is tractable. The paper characterizes this as the "low-order" case — exact or near-exact computation when the number of features is manageable.

##### Max SHAP

The max function — $f(x) = \max(x_1, x_2, \ldots, x_M)$ — is important in practice because max pooling is a standard operation in convolutional neural networks, and understanding how credit flows through a max operation is necessary for attributing predictions to input features in image models. Computing Shapley values for the max function naively requires $O(M \cdot 2^M)$ evaluations (for each feature, evaluate on all $2^{M-1}$ subsets of other features). The paper presents an algorithm that reduces this to $O(M^2)$.

**The algorithm.** The paper states the approach concisely: "Using a permutation formulation of Shapley values, we can calculate the probability that each input will increase the maximum value over every other input. Doing this on a sorted order of input values lets us compute the Shapley values of a max function with $M$ inputs in $O(M^2)$ time instead of $O(M2^M)$."

The full algorithm is in the paper's Supplementary Material, but the intuition is clear from this description. In any random ordering (permutation) of the $M$ inputs, the max function's output is determined by the *largest value seen so far* as features are added one by one. When feature $i$ is added, its marginal contribution is $\max(0, x_i - \text{current_max})$ — it increases the running maximum only if its value exceeds all previously added values. The probability that feature $i$ increases the maximum depends on how many features have values larger than $x_i$ and where they appear in the ordering relative to $i$. By sorting the inputs by value and computing these probabilities using combinatorial counting, the Shapley values can be computed without enumerating all subsets.

**Why this matters for Deep SHAP.** Max SHAP provides the exact Shapley values for the max function, which Deep SHAP (described next) can plug in as a component-level solution. Instead of using DeepLIFT's heuristic rule for attributing through max pooling, Deep SHAP uses the actual Shapley values, which — as the paper's user study shows (Figure 4B) — produce explanations that align with human intuition on max-function problems.

##### Deep SHAP (DeepLIFT + Shapley Values)

Deep SHAP is the paper's method for efficiently approximating SHAP values for deep neural networks by exploiting their compositional structure. It adapts DeepLIFT's back-propagation framework but replaces DeepLIFT's heuristic linearization rules with Shapley-value-based linearizations for each network component.

**The core insight.** DeepLIFT's "summation-to-delta" property and its compositional back-propagation are structurally compatible with Shapley values. If we can compute exact or approximate SHAP values for each simple component in a deep network (linear layers, activation functions, pooling operations, element-wise operations), we can compose them through the network using a chain rule, propagating SHAP values from the output layer back to the input, just as gradients are back-propagated during training.

**The connection between DeepLIFT's reference value and SHAP.** The paper observes: "If we interpret the reference value in Equation 3 as representing $\mathbb{E}[x]$ in Equation 12, then DeepLIFT approximates SHAP values assuming that the input features are independent of one another and the deep model is linear." This is a crucial interpretive move. DeepLIFT's reference input $r$ plays the same role as $\mathbb{E}[x]$ in the SHAP framework under the feature independence + model linearity approximation: it is the "baseline" value that a feature is compared against. By recognizing this, the paper can treat DeepLIFT as an approximation of SHAP values and then improve that approximation by replacing heuristic component linearizations with Shapley-value-derived ones.

**The Deep SHAP back-propagation rule (Equations 13–16).** The paper defines a recursive procedure for composing SHAP values through a network. The notation uses multipliers, defined as the SHAP value divided by the feature's deviation from its expected value:

For a simple component $f_3$ that takes inputs from $f_1$ and $f_2$ (as in Figure 2B):

$$m_{x_j f_3} = \frac{\phi_i(f_3, x)}{x_j - \mathbb{E}[x_j]} \quad \forall j \in \{1, 2\}$$

where $m_{x_j f_3}$ is the multiplier representing how much a unit change in input $x_j$ changes the SHAP value attributed through $f_3$. These multipliers are computed for each simple component using analytically solved or numerically estimated SHAP values for that component alone.

For the outputs of $f_1$ and $f_2$ (which feed into $f_3$):

$$m_{y_i f_j} = \frac{\phi_i(f_j, y)}{y_i - \mathbb{E}[y_i]}$$

where $y$ is the output of $f_j$ and $y_i$ is the $i$-th element of that output (for vector-valued components).

The chain rule composes these multipliers:

$$m_{y_i f_3} = \sum_{j=1}^{2} m_{y_i f_j} \cdot m_{x_j f_3}$$

This is a standard chain rule: the effect of changing $y_i$ (an intermediate activation) on the final output SHAP value is the sum over all downstream components $f_j$ that consume $y_i$, of (effect of $y_i$ on $f_j$) × (effect of $f_j$'s output on $f_3$'s output). For a deep network, this sum generalizes to all forward connections from a given activation to the output.

Finally, the SHAP value for the original input features is approximated linearly:

$$\phi_i(f_3, y) \approx m_{y_i f_3} (y_i - \mathbb{E}[y_i])$$

**What this does operationally.** Deep SHAP traverses the network from output to input once, computing and composing multipliers. At each component, it needs the SHAP values for that component in isolation — but since the components are simple (linear, element-wise activation, max pool), their SHAP values can be pre-computed analytically (using Linear SHAP for linear layers, Max SHAP for max pooling) or via fast numerical integration (for activation functions with one input, where the Shapley value computation is trivial). The full pass has complexity proportional to a standard forward/backward pass, making it vastly cheaper than Kernel SHAP (which would require thousands of forward passes for a deep network).

**Why this is an improvement over original DeepLIFT.** Original DeepLIFT uses heuristic rules to define how each component's output change is attributed to its inputs. For example, the "RevealCancel" rule for element-wise multiplication, or the rule for max pooling. These rules were designed to be reasonable but were not derived from a consistent principle. Deep SHAP replaces these heuristics with the actual Shapley values for each component type. Because Shapley values satisfy consistency (Property 3) at the component level, and the linear composition rule preserves this property (approximately, due to the linearization assumption), the resulting attributions better approximate the true SHAP values for the full model.

**The max pooling example.** This is where the improvement is clearest. Original DeepLIFT needed a heuristic rule for attributing the max function's output to its inputs. Deep SHAP uses Max SHAP (the $O(M^2)$ exact algorithm) to compute the correct Shapley values for each max pooling operation, and then propagates those through the network. The user study (Figure 4B) validates that this produces attributions matching human intuition on a standalone max-function problem, which the paper presents as addressing "the open problem of max pooling functions in DeepLIFT."

**Limitations of the approximation.** Deep SHAP inherits the assumptions of Equation 11 (feature independence) and Equation 12 (model linearity). The feature independence assumption means that when evaluating $\mathbb{E}[f(z) \mid z_S]$, the unknown features are treated as independent of the known features. The model linearity assumption means that non-linear components are approximated as linear around their operating point (using the multipliers $m$). These are the same assumptions original DeepLIFT makes — Deep SHAP improves *what* linearization is used for each component (Shapley-value-based rather than heuristic), but does not escape the fact that it is still linearizing a non-linear model. For networks with strong non-linearities or feature interactions that cannot be captured by the linear approximation, Deep SHAP's values will diverge from the true SHAP values (which would require the expensive conditional expectation computation).

---

#### Summary of Design Choices and Their Justifications

- **Explanation model as linear function of binary variables (Definition 1):** Chosen because it unifies six existing methods, enables the Shapley uniqueness result, and produces human-comprehensible explanations (each feature gets one number, numbers sum to the prediction).

- **Local accuracy, missingness, and consistency as the three axioms (Properties 1–3):** Chosen because they map to established Shapley axioms (efficiency, dummy, monotonicity) and because violations produce demonstrably unintuitive attributions (as shown in user studies). Local accuracy ensures the explanation is self-consistent; missingness prevents nonsensical attributions to absent features; consistency ensures that uniformly more important features receive uniformly larger attributions.

- **Conditional expectations for handling absent features (Equation 9):** Chosen over marginal expectations because conditioning on known features is more faithful to the data-generating process. The paper makes the assumptions needed for tractable approximation (independence, linearity) explicit rather than implicit.

- **Kernel SHAP as regression-based Shapley estimation (Theorem 2):** Chosen because joint regression estimation is more sample-efficient than independent per-feature Shapley sampling, and because it reveals exactly how LIME's heuristic kernel deviates from the Shapley-optimal kernel.

- **Debiased lasso for finite-sample Kernel SHAP:** Chosen as a practical compromise — lasso regularization prevents overfitting when only a sample of $z'$ vectors is used, and debiasing removes the L1 shrinkage bias so coefficients approximate the true Shapley values.

- **Deep SHAP's compositional back-propagation:** Chosen because it achieves computational efficiency for deep networks by exploiting compositional structure, at the cost of accepting the feature independence and model linearity assumptions. The improvement over original DeepLIFT comes from replacing heuristic component linearizations with Shapley-value-derived ones.

## 4. Key Insights and Innovations

### Innovation 1: Explanations Are Models — The Explanation Model Perspective Reframes the Entire Problem

Prior to this work, the field of interpretable machine learning operated with a fragmented vocabulary. Methods were described by their procedures: LIME "perturbs inputs and fits a local linear model," DeepLIFT "back-propagates activation differences," Shapley sampling "averages over feature subsets." Each paper introduced its own framing, making it difficult to compare methods or reason about what properties a good explanation should satisfy.

The paper's foundational conceptual move is to **treat every explanation as itself a model** — what the authors call the "explanation model" $g$. This is not a technical trick; it is a reframing that changes what questions you can ask. Once you recognize that LIME, DeepLIFT, layer-wise relevance propagation, and Shapley-based methods all produce explanations that fit the same functional form — a linear function of binary variables, $g(z') = \phi_0 + \sum_i \phi_i z_i'$ — you can stop asking "which method is better?" and start asking "which choice of $\phi_i$ values is correct under what desiderata?" The former is a benchmarking question without a theoretical answer; the latter is a constrained optimization problem with a unique solution.

This reframing is intellectually distinctive because it **abstracts away implementation details to reveal structural identity**. The fact that LIME (which samples perturbed inputs and runs weighted least squares) and Shapley regression (which retrains models on all feature subsets) both produce coefficients in the same additive form is not obvious from their procedural descriptions. The paper's Definition 1 makes this identity explicit and, in doing so, creates a shared mathematical language for a previously balkanized literature.

The move is fundamental, not incremental. Prior work on Shapley values for model interpretation (Lipovetsky and Conklin, 2001; Štrumbelj and Kononenko, 2014) treated Shapley values as one technique among many, without recognizing that apparently unrelated methods were approximating the same quantity. LIME's authors (Ribeiro et al., 2016) presented their approach as a novel local approximation framework, not as a specific estimation procedure for Shapley values. The paper's unification reveals that **these were never genuinely different explanation paradigms — they were different estimation strategies for the same underlying explanation model class, diverging primarily in how faithfully they approximate the Shapley solution**.

The evidence for this unification is not a single figure but the entire structural argument of Section 2, where each of the six methods is shown to produce explanations matching Equation 1 with a specific choice of the simplified input mapping $h_x$. The paper's ability to map DeepLIFT's "summation-to-delta" property (which was presented by its authors as a distinctive feature of their method) directly onto the additive attribution form — with $\phi_0 = f(r)$ and $\phi_i = C_{\Delta x_i \Delta o}$ — is the kind of connection that only becomes visible once you have the right abstraction.

### Innovation 2: The Three Axioms Provide a Normative Theory of Explanation Quality — Not Just a Unification, but a Filter

Unifying existing methods under a shared form is a taxonomic contribution. But the paper goes further: it provides a **normative theory** that distinguishes better explanations from worse ones *within* that form. The three properties — local accuracy, missingness, and consistency — are not presented as arbitrary desiderata but as necessary and sufficient conditions for uniqueness (Theorem 1). This transforms the landscape from "many methods exist, pick one" to "there is exactly one explanation in this class that satisfies these properties; any method producing different values necessarily violates at least one of them."

What makes this intellectually distinctive is that the properties are **operationally falsifiable**. You can take any additive feature attribution method, check whether its attributions sum to the prediction (local accuracy), check whether absent features receive non-zero attributions (missingness), and check whether uniformly increasing a feature's marginal contribution can paradoxically decrease its attribution (consistency). The paper demonstrates that both LIME and original DeepLIFT fail at least one of these checks — not as a theoretical possibility but as an empirical fact with visible consequences in user studies (Figure 4).

The consistency property (Property 3) deserves particular attention because it is the least obvious of the three. Local accuracy is intuitive — an explanation should be self-consistent. Missingness is bookkeeping — absent features shouldn't get credit. But consistency captures a deeper notion of **monotonicity under model change**: if you modify the model so that a feature becomes uniformly more important (its marginal contribution never decreases for any subset of other features), its attribution should not go down. This property rules out attribution schemes where the importance assigned to a feature depends on factors other than that feature's actual marginal contributions — for instance, on the particular heuristics chosen for linearizing non-linear components (as in original DeepLIFT) or on the arbitrary choice of weighting kernel (as in LIME).

This is a fundamental theoretical advance rather than an incremental refinement, because it **imports an established axiomatic characterization from cooperative game theory (Young, 1985) into machine learning interpretability** and uses it to evaluate methods that were developed without awareness of these axioms. The paper shows that Young's proof — which established Shapley values as the unique solution satisfying efficiency, symmetry, and monotonicity in cooperative games — maps directly onto the additive feature attribution setting, with one axiom (symmetry) becoming redundant and a new one (missingness) being added to handle the binary input representation.

The evidence that this matters beyond theory comes from the user studies (Section 5.2, Figure 4). On the fever-and-cough sickness scoring problem (Figure 4A), where the model output depends on whether exactly one symptom is present, SHAP values attribute credit in a way that matches the most common human explanation, while LIME produces a qualitatively different allocation. On the max-function profit allocation (Figure 4B), SHAP values again match human intuition while original DeepLIFT diverges. These are not high-dimensional, ambiguous cases — they are toy problems where the "correct" attribution is intuitively clear, and methods that violate the axioms produce the wrong answer. The paper's ability to predict *which* methods will fail *which* problems based on axiom violation is what elevates the axioms from mathematical formalism to practical diagnostic tools.

### Innovation 3: Kernel SHAP Reveals That Shapley Values Are a Regression Problem — and LIME's Heuristic Kernel Is the Wrong One

The paper's most surprising technical insight is that Shapley values — which had been understood for decades as a combinatorial average over feature subsets — can be computed exactly by solving a **specific weighted linear regression problem** (Theorem 2). This is not an approximation; it is an equivalence. The Shapley kernel $\pi_{x'}(z') = \frac{M-1}{\binom{M}{|z'|}|z'|(M-|z'|)}$ makes the least-squares regression coefficients equal to the Shapley values when all $2^M$ simplified inputs are included in the regression.

This is intellectually distinctive because it **bridges two literatures that had no awareness of each other**: the game-theoretic Shapley value literature (focused on cooperative games and fair allocation) and the local approximation literature (focused on model-agnostic interpretability via perturbation-based regression). The connection is non-obvious — Shapley values are typically motivated by axioms about fair division among players with complementary contributions, while LIME is motivated by the idea that complex models are approximately linear in a local neighborhood. Theorem 2 shows that these motivations are compatible only when the "neighborhood" is weighted with a specific, combinatorially-derived kernel that has nothing to do with spatial proximity.

The comparison with LIME's heuristic kernel (Figure 2A) makes the innovation concrete. LIME weights simplified inputs by their distance from the all-present vector $x'$ — points close to the original input get high weight, points far away get low weight. This is intuitively natural: you want your local approximation to be accurate near the point you're explaining. But the Shapley kernel is **U-shaped**: it assigns highest weight to simplified inputs with very few features present (small $|z'|$) and to those with nearly all features present (large $|z'|$), with lower weight in the middle. There is no notion of "distance" in the Shapley kernel — all simplified inputs with the same number of present features get identical weight regardless of *which* features are present.

This U-shape has a deep combinatorial justification that the distance-based heuristic misses. In the Shapley averaging formula (Equation 8), marginal contributions for small coalitions (where few other features are present) and large coalitions (where almost all other features are present) receive higher weight because there are fewer ways to arrange such coalitions in the permutation averaging. The Shapley kernel ensures that the regression's least-squares solution reproduces exactly this weighting structure. LIME's distance-based kernel, by overweighting points near the all-present vector, systematically biases attributions toward features that happen to matter in large-coalition contexts and away from features that matter primarily in small-coalition contexts.

This is a fundamental insight, not an incremental refinement of LIME. The paper is not proposing a "better LIME kernel" — it is showing that the regression approach LIME pioneered can be made *exact* rather than heuristic, and that doing so reveals a kernel that is qualitatively different from what anyone would have guessed based on spatial intuition alone. The empirical consequence (Figure 3) is that Kernel SHAP converges to the true Shapley values with substantially fewer model evaluations than classical Shapley sampling, particularly when the true model is sparse (depends on only a few of the available features). The regression formulation's joint estimation of all $\phi_i$ shares information across features, making it sample-efficient in a way that independent per-feature sampling is not.

### Innovation 4: Deep SHAP Transforms DeepLIFT from Heuristic to Principled by Replacing Rules with Shapley Values

Original DeepLIFT (Shrikumar et al., 2016, 2017) was a significant practical advance: it enabled fast attribution for deep networks by back-propagating "multipliers" that linearize each network component. But the linearization rules — how to attribute a max pooling operation's output to its inputs, how to handle element-wise multiplication, what reference value to use — were chosen heuristically. The paper's Deep SHAP method replaces these heuristic rules with **Shapley-value-derived linearizations** for each component, using the same compositional back-propagation framework.

The intellectual move here is to recognize that **component-level Shapley values compose through a network via a chain rule** (Equations 13–16). This is not obvious: Shapley values are defined globally (averaging over all feature subsets for the entire model), so there is no inherent reason they should be computable by local, layer-by-layer composition. The paper shows that if you accept the feature independence and model linearity assumptions (Equations 11–12), then Shapley values *do* compose linearly, and the composition rule is precisely a Shapley-value version of the gradient chain rule. The multiplier $m_{x_j f_3} = \frac{\phi_i(f_3, x)}{x_j - \mathbb{E}[x_j]}$ plays the same structural role as a partial derivative in back-propagation, but is derived from Shapley values rather than from instantaneous gradients.

This is an incremental advance in the sense that it builds directly on DeepLIFT's architecture — the back-propagation structure, the use of multipliers, the summation-to-delta property — but the improvement is conceptually fundamental: it replaces *ad hoc* engineering choices with the unique solution that satisfies the three Shapley axioms at the component level. The max pooling case is the clearest demonstration. Original DeepLIFT needed a rule for attributing max outputs; the authors picked something reasonable but acknowledged it as an open problem (Shrikumar et al., 2017). Deep SHAP uses Max SHAP — an $O(M^2)$ exact algorithm for the max function's Shapley values (described in Section 4.2 and the Supplementary Material) — and plugs it into the compositional framework. The user study (Figure 4B) confirms that this produces attributions matching human judgment on a max-function profit allocation task where original DeepLIFT diverges.

The broader significance is methodological: Deep SHAP demonstrates that **the Shapley framework is not just an evaluative standard but a constructive tool**. You don't have to choose between using Shapley values (principled but expensive) and using fast heuristics (DeepLIFT, LRP). You can compute Shapley values for simple components analytically (Linear SHAP for linear layers, Max SHAP for pooling) and compose them, inheriting the axioms at the component level and approximately preserving them at the full-model level. This opens the door to a family of model-specific SHAP approximations that trade off exactness for speed, all anchored to the same axiomatic foundation.

The evidence for Deep SHAP's improvement appears in Figure 5, which extends DeepLIFT's own convolutional network example on MNIST digit classification. When pixels are masked in order of attributed importance to switch a prediction from class 8 to class 3, the "New DeepLIFT" (closer to SHAP values) produces a larger and more consistent change in log-odds than original DeepLIFT across 20 random images. The SHAP-computed attributions identify pixels that are more causally relevant to the class decision, demonstrating that the theoretical improvement (satisfying consistency at the component level) translates to practical benefit (better identification of decision-relevant input regions).

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper does not use a single standardized benchmark dataset for all experiments. Instead, experiments span three distinct domains: (1) synthetic decision tree models with 10 or 100 features for computational efficiency comparisons (Figure 3); (2) human-designed explanation tasks for user studies (Figure 4) — a sickness scoring model with two binary symptoms (fever, cough) and a max-function profit allocation with three inputs (scores of 5, 4, and 0); and (3) the MNIST handwritten digit dataset (LeCun et al., 1998) for the convolutional network explanation comparison (Figure 5), using the same pre-trained model and example inputs as Shrikumar et al. (2017). The diversity of evaluation settings is deliberate: each targets a different claim — computational efficiency, alignment with human intuition, and practical utility for deep network explanations, respectively.

- **Base model(s).** Three distinct model types are used across experiments. For the computational efficiency comparison (Figure 3), the paper uses decision tree models — one "dense" tree using all 10 input features and one "sparse" tree using only 3 of 100 input features, both evaluated for a single input. For the user studies (Figure 4), the models are simple logical functions: a sickness score model where output = 2 when both fever and cough are present, 5 when exactly one is present, and 0 when neither is present; and a max function where profit equals the maximum score among three individuals. For the MNIST experiments (Figure 5), the paper uses a convolutional neural network with two convolution layers, two dense layers, and a 10-way softmax output, identical to the model used in Shrikumar et al. (2017). The choice of models spans from trivially interpretable (where ground-truth attributions are known by construction) to practically complex (where attributions must be validated indirectly), allowing the paper to test both whether methods recover known-correct answers and whether they produce useful results in realistic settings.

- **Metrics.** Three distinct evaluation metrics are used, each matched to the experiment's goal:
  - **Convergence to true Shapley values (Figure 3):** For the synthetic decision tree experiments, the "true Shapley value" is computed exactly via enumeration of all feature subsets (Equation 8). Feature importance estimates from each method are plotted against this ground truth as the number of model evaluations increases. The 10th and 90th percentiles across 200 replicate estimates at each sample size quantify estimation variance.
  - **Agreement with human explanations (Figure 4):** For the user studies conducted on Amazon Mechanical Turk, the metric is the modal (most common) human explanation among 30 participants (sickness task, Figure 4A) and 52 participants (profit allocation task, Figure 4B). Participants were asked to "assign credit for the output among the inputs," and the paper compares each method's attribution vector to the human modal attribution. The paper reports bar charts showing the attribution values assigned by each method side-by-side with the human modal explanation; no quantitative agreement score (e.g., correlation, mean absolute error) is reported.
  - **Change in log-odds under masking (Figure 5B):** For the MNIST experiment, the paper measures how the model's predicted log-odds for a target class change when pixels are masked in order of attributed importance. Specifically, 20% of pixels are selected for masking based on each method's attribution for switching from class 8 to class 3, and the resulting change in the log-odds ratio between these two classes is plotted (mean and standard deviation over 20 random images). A larger change indicates that the attributed pixels are more causally relevant to the model's class decision.

- **Baselines.** The paper compares against three existing methods, each representing a different point in the additive feature attribution landscape:
  - **LIME** (Ribeiro et al., 2016): The open-source implementation, used as a representative of local linear approximation methods with heuristic kernel weighting. Evaluated in the computational efficiency comparison (Figure 3), the user studies (Figure 4), and the MNIST experiment (Figure 5). For the MNIST experiment, LIME was modified to use single-pixel segmentation over digit pixels and run with 50k samples to improve performance (Supplementary Figure 1).
  - **Shapley sampling values** (Štrumbelj and Kononenko, 2014): The sampling-based approximation to classic Shapley value equations, used as a baseline in the computational efficiency comparison (Figure 3). This method estimates each feature's Shapley value independently using a permutation-based sampling approximation.
  - **Original DeepLIFT** (Shrikumar et al., 2016, 2017): The heuristic-rule-based back-propagation method, used as a baseline in the user studies (Figure 4B, max-function task) and the MNIST experiment (Figure 5). The paper refers to "Orig. DeepLIFT" as the version without explicit Shapley approximations, and "New DeepLIFT" as the version updated (in Shrikumar et al., 2017) to better approximate Shapley values.

- **Generation budget / compute accounting.** Compute is measured in **evaluations of the original model function** — that is, the number of times $f_x(z')$ is computed for different simplified inputs $z'$. This is the dominant cost for model-agnostic methods since each evaluation requires a forward pass through the original model (or retraining in the case of Shapley regression values). For the computational efficiency comparison (Figure 3), the x-axis shows the number of model evaluations increasing from 0 to approximately 2000 (dense model, Figure 3A) and 0 to approximately 6000 (sparse model, Figure 3B). For Kernel SHAP, the number of evaluations equals the number of sampled $z'$ vectors used in the weighted regression. For Shapley sampling, each feature's estimate requires separate sampling, so the total evaluations equals the sum of samples across all features. For Deep SHAP (used in Figure 5), the cost is a single backward pass through the network — comparable to one training iteration — rather than many forward passes, but this efficiency gain is not directly compared to model-agnostic methods in terms of evaluation count. The paper does not provide a FLOPs comparison between Kernel SHAP and Deep SHAP.

- **Cross-validation / statistical protocol.** The experiments do not employ traditional cross-validation since they do not involve training models on data splits. Instead, the paper uses two forms of statistical quantification:
  - **Replicate estimates (Figure 3):** For the computational efficiency comparison, 200 replicate estimates are performed at each sample size, and the 10th and 90th percentiles of the estimated feature importance for one feature are plotted to show estimation variance. This quantifies how reliably each method converges to the true Shapley value under random sampling of simplified inputs.
  - **Aggregation over images (Figure 5B):** For the MNIST masking experiment, the change in log-odds is averaged over 20 random images from the digit dataset, with standard deviation error bars shown. This captures whether the observed improvement of SHAP-aligned attributions over original DeepLIFT holds consistently across different inputs rather than being driven by a single favorable example.

### Main Quantitative Results

#### Computational Efficiency: Kernel SHAP vs. Shapley Sampling vs. LIME

The paper's central computational claim is that **Kernel SHAP converges to the true Shapley values with substantially fewer model evaluations than classical Shapley sampling**, and that **LIME values diverge from the Shapley solution** even at high sample sizes (Figure 3). The comparison is performed on two synthetic decision tree models, with feature importance tracked for a single feature as the number of model evaluations grows.

**Dense model (Figure 3A):** A decision tree using all 10 of 10 input features is explained for a single input. The true Shapley value for the tracked feature (computed by exact enumeration) is shown as a horizontal dashed line. Kernel SHAP (using debiased lasso) converges to the true value by approximately 200–400 model evaluations, with tight 10th–90th percentile bands indicating low variance across replicates. Shapley sampling values converge more slowly, requiring roughly 1000–1500 evaluations to reach comparable accuracy, and show wider percentile bands (higher variance) at intermediate sample sizes. LIME values, using the open-source implementation with heuristic kernel, converge to a value that is visibly offset from the true Shapley value — the estimate stabilizes but at the wrong number, indicating systematic bias rather than merely high variance. The paper states this demonstrates that "values from LIME can differ significantly from SHAP values that satisfy local accuracy and consistency."

**Sparse model (Figure 3B):** A decision tree using only 3 of 100 input features is explained for a single input. The difference between methods is more pronounced in this high-dimensional sparse setting. Kernel SHAP converges to the true Shapley value (which is zero for the tracked feature if it is one of the 97 unused features, or non-zero if it is one of the 3 used features — the paper does not specify which is tracked, but the estimate converges near zero in the figure) within approximately 1000–2000 evaluations. Shapley sampling shows dramatically higher variance, with the 10th–90th percentile band spanning a wide range even at 6000 evaluations, indicating unreliable estimates. LIME estimates again stabilize at a systematically biased value distinct from the true Shapley value. The paper interprets this as evidence that Kernel SHAP's joint regression estimation is particularly advantageous in sparse settings, where "the true model depends on only a few features" — the regression formulation shares information across features, effectively regularizing estimates for the many zero-importance features.

**Quantitative summary of Figure 3:** The paper does not report exact accuracy numbers (mean absolute error from true Shapley value at specific evaluation counts), relying instead on the visual comparison of convergence trajectories. The qualitative pattern is unambiguous: Kernel SHAP reaches the true value faster and with lower variance than Shapley sampling, while LIME converges to a biased estimate. The performance gap between Kernel SHAP and Shapley sampling widens as the feature space becomes larger and sparser (10 features → 100 features), suggesting that the advantage of joint regression estimation grows with dimensionality.

#### Consistency with Human Intuition: SHAP vs. LIME vs. DeepLIFT

The paper's second major empirical claim is that **SHAP values produce explanations more consistent with human judgments than methods that violate Properties 1–3**, tested on two simple logical models where the "correct" attribution is intuitively clear (Figure 4).

**Sickness scoring task (Figure 4A):** The model outputs a sickness score based on two binary symptoms (fever, cough): score = 2 when both present, 5 when exactly one is present, 0 when neither present. The prediction being explained has fever and cough both present, so $f(x) = 2$. Thirty Mechanical Turk participants were asked to assign credit for the score of 2 between the two symptoms. The modal human explanation assigns approximately +2 to fever and 0 to cough (the bar chart in Figure 4A shows the human bar with these approximate values). SHAP values produce attributions that closely match this pattern (the paper's bar chart shows SHAP assigning positive credit to both symptoms but with a distribution matching human judgment). LIME produces a qualitatively different attribution pattern — the paper's bar chart shows LIME assigning substantially different values to the two symptoms compared to human judgments. The paper interprets LIME's divergence as a consequence of violating local accuracy and/or consistency: LIME's heuristic kernel and regularization prevent its attributions from satisfying the axioms that produce the intuitive allocation.

**Max-function profit allocation (Figure 4B):** Three individuals earn profit based on the maximum number of questions any of them answers correctly: Person 1 gets 5 right, Person 2 gets 4 right, Person 3 gets 0 right, so total profit = $5. Fifty-two Mechanical Turk participants allocated credit for the $5 among the three individuals. The modal human explanation attributes all $5 to Person 1 (the one with the maximum score), with $0 to Persons 2 and 3. SHAP values assign credit in a way that "closely matches" this human modal explanation (the paper's bar chart shows SHAP attributions predominantly to Person 1, with small non-zero values to the others). Original DeepLIFT produces a qualitatively different allocation — the paper's Figure 4B shows original DeepLIFT assigning substantial credit to Persons 2 and 3, which disagrees with human judgment. The paper frames this as directly addressing "the open problem of max pooling functions in DeepLIFT" — DeepLIFT's heuristic linearization of the max function produced unintuitive attributions, while SHAP's Max SHAP algorithm (the $O(M^2)$ exact computation) recovers attributions matching human intuition.

**Quantitative limitations of the user study reporting.** The paper reports these results as bar charts showing per-method attribution vectors alongside the human modal explanation, but does not report a quantitative agreement metric (e.g., correlation coefficient, mean squared error, or statistical test for difference from human judgments). The sample sizes (30 and 52 participants) are small by contemporary standards for user studies, and only the modal (most common) explanation is shown — the distribution of human responses (e.g., what fraction of participants gave each attribution pattern) is not reported. This makes it difficult to assess whether SHAP's agreement with human judgment is statistically robust or whether alternative reasonable attribution schemes (e.g., splitting credit between Persons 1 and 2 in the max task) were common among participants but not captured by the modal summary.

#### Explaining Class Differences in MNIST Digit Classification

The third empirical claim is that **better approximation of SHAP values produces more causally relevant feature attributions for deep neural networks**, tested by comparing attribution-guided pixel masking on MNIST (Figure 5).

**Masking experiment design (Figure 5A).** The paper uses a pre-trained convolutional network on MNIST, identical to the model used in Shrikumar et al. (2017). An input image of a digit 8 is explained, and each method produces a saliency map indicating which pixels increase or decrease the probability of class 8 vs. class 3. The "Masked" row in Figure 5A shows the result of masking (setting to zero or a neutral value) 20% of the pixels chosen to switch the prediction from class 8 to class 3, using attributions from each method. The paper states: "Red areas increase the probability of that class, and blue areas decrease the probability." The visual comparison in Figure 5A shows that SHAP, LIME, and New DeepLIFT (the updated version) produce qualitatively similar saliency maps that highlight the regions distinguishing an 8 from a 3 (the left side of the top loop and the middle crossbar), while Original DeepLIFT highlights somewhat different regions.

**Quantitative masking results (Figure 5B).** The change in log-odds between class 8 and class 3 after masking 20% of pixels is measured across 20 random images, with mean and standard deviation plotted. Original DeepLIFT achieves a mean change in log-odds of approximately 30–35 (reading from the bar chart). New DeepLIFT achieves a larger change, approximately 40–45. SHAP (computed using Kernel SHAP with 50k samples) achieves a similar or slightly larger change, approximately 40–50. LIME (modified to use single-pixel segmentation, 50k samples) achieves a change comparable to SHAP, approximately 40–50. The error bars (standard deviation) overlap substantially among SHAP, LIME, and New DeepLIFT, while Original DeepLIFT's error bar lies mostly below the others, suggesting the improvement is modest but consistent.

**Interpretation and limitations.** The paper interprets these results as showing that "better estimates of SHAP values" produce superior attributions, citing the increased log-odds change for New DeepLIFT over Original DeepLIFT and the comparable performance of SHAP and LIME. However, Figure 5B does not show a clear SHAP > New DeepLIFT > LIME > Original DeepLIFT hierarchy — SHAP, LIME, and New DeepLIFT are all within overlapping error bars, while Original DeepLIFT is lower. The experiment thus primarily demonstrates that **heuristic DeepLIFT (original) underperforms everything else**, rather than that SHAP specifically outperforms LIME or that improving DeepLIFT's approximation to SHAP produces monotonic improvement. The fact that LIME (which the paper showed produces biased estimates in Figure 3) performs comparably to SHAP on this metric suggests that the masking-based evaluation may not be sensitive to the specific attribution differences that the axioms guarantee. A possible explanation: for this particular model and class pair, many reasonable attribution methods identify similar sets of discriminative pixels, and the masking test measures coarse-grained causal relevance rather than fine-grained attribution accuracy.

### Ablation Studies and Robustness Checks

The paper does not contain formal ablation studies in the modern sense (systematically removing or varying components of the proposed method and measuring the impact). The empirical validation is instead structured as comparative evaluation against baseline methods under different conditions. Several comparisons serve a function analogous to ablation studies by isolating the effect of specific design choices:

- **Heuristic LIME kernel vs. Shapley kernel (Figure 2A vs. Figure 3):** The comparison of LIME's heuristic kernel with the Shapley kernel is not isolated as an ablation — the full LIME implementation (with its loss, kernel, and regularization) is compared against Kernel SHAP (with the Shapley kernel and no regularization). The convergence gap in Figure 3 demonstrates that LIME's design choices collectively produce biased estimates, but the experiment does not disentangle whether the bias comes from the kernel, the regularization, or the loss function. A clean ablation would test LIME with the Shapley kernel and no regularization to see whether the kernel alone accounts for the difference.

- **Original DeepLIFT vs. New DeepLIFT (Figure 5):** This comparison isolates the effect of moving DeepLIFT's component linearization rules closer to Shapley values. Original DeepLIFT uses the heuristic rules from Shrikumar et al. (2016); New DeepLIFT incorporates updates from Shrikumar et al. (2017) that "better match Shapley values." The improvement in log-odds change (Figure 5B) suggests that better Shapley approximation improves attribution quality, supporting the paper's theoretical argument. However, Deep SHAP (the paper's proposed compositional method using exact component Shapley values) is not directly compared to New DeepLIFT in this experiment — SHAP in Figure 5 is computed using Kernel SHAP (model-agnostic regression), not Deep SHAP. So the experiment validates the value of Shapley approximation but does not directly benchmark Deep SHAP.

- **Dense vs. sparse model (Figure 3A vs. 3B):** This comparison assesses how the methods' relative performance changes with feature space dimensionality and sparsity. Kernel SHAP's advantage over Shapley sampling widens in the sparse 100-feature case (Figure 3B), suggesting that joint regression estimation is particularly beneficial when most features have zero true importance. This functions as a robustness check across problem characteristics, though only two synthetic configurations are tested.

- **Model-agnostic SHAP (Kernel SHAP) vs. compositional SHAP approximation (Deep SHAP):** The paper does not directly compare these two estimation approaches for SHAP values. Kernel SHAP is used for the MNIST experiment (Figure 5, requiring 50k model evaluations per image), while Deep SHAP is described algorithmically but its computational cost and approximation accuracy relative to Kernel SHAP are not empirically quantified. This is a significant gap: a central claim of Section 4.2 is that Deep SHAP enables fast compositional approximation, but the paper provides no experiment showing that Deep SHAP's values approximate Kernel SHAP's values for the same model, or quantifying the speed-accuracy trade-off.

- **Choice of conditional vs. marginal expectation for absent features (Equations 9–12):** The paper does not empirically compare SHAP values computed under the conditional expectation definition (Equation 9) vs. the feature independence approximation (Equation 11) vs. the model linearity approximation (Equation 12). All experiments use one of the approximations (e.g., the user study tasks are deterministic logical functions where independence and linearity assumptions are not relevant; the MNIST experiment with Kernel SHAP likely uses the feature independence or linearity approximation for computational tractability, but this is not specified). The impact of these approximation choices on attribution quality is left as an open question.

### Critical Assessment

#### Does SHAP produce explanations that are more consistent with human intuition than LIME and DeepLIFT?

The user study (Figure 4) provides direct evidence for two very specific toy problems. On the sickness scoring task (Figure 4A), SHAP matches the modal human explanation while LIME diverges — supporting the claim that violating the axioms (as LIME does) produces unintuitive attributions. On the max-function task (Figure 4B), SHAP matches human intuition while original DeepLIFT diverges — supporting the claim that heuristic linearization of non-linear components (as in original DeepLIFT) produces unintuitive attributions, and that Shapley-value-based linearization (Max SHAP) fixes this.

**What the experiment does not demonstrate:** The two test cases are extremely simple (2 binary features, 3 scalar inputs) and the models are fully transparent logical functions. It is not demonstrated that SHAP's advantage over LIME or DeepLIFT persists for models with more features, continuous inputs, or non-linear interactions — precisely the complex model settings where interpretability tools are needed in practice. The human study uses only the modal explanation as a target; if human participants showed substantial disagreement about correct attributions (which is plausible even for these simple tasks), the modal explanation may not represent a consensus "ground truth." The sample sizes (30 and 52) are small, and no demographic or qualification details about the Mechanical Turk participants are provided. The study does not test whether the observed differences between methods are *perceptible* to users — that is, whether users looking at SHAP vs. LIME explanations would make better decisions, trust the model more, or identify errors faster. It only tests whether the attributions match what users *say* the attribution should be for a model they fully understand.

#### Is Kernel SHAP more sample-efficient than Shapley sampling?

Figure 3 provides clean evidence for this claim on two synthetic decision tree models with 10 and 100 features. Kernel SHAP converges to the true Shapley value with visibly fewer model evaluations and lower variance. The advantage is larger in the sparse 100-feature case, which is practically important since many real-world models use large feature sets but depend strongly on only a subset.

**What the experiment does not demonstrate:** The comparison uses only decision tree models. It is not shown whether the advantage holds for other model types (neural networks, kernel methods, ensembles) where the relationship between features and output has different structure. The experiment tracks only one feature's importance estimate; it does not show whether the joint convergence (all features simultaneously) is faster for Kernel SHAP, though the regression formulation theoretically supports this. The "true Shapley value" is computed by exact enumeration, which is only feasible for these small toy models — for larger models where Kernel SHAP would actually be used, there is no ground truth to validate against, so the convergence trajectory is unknowable.

#### Does better Shapley approximation produce better attributions for deep networks?

Figure 5 provides partial support. Original DeepLIFT (worst Shapley approximation) underperforms New DeepLIFT, SHAP, and LIME on the masking-based log-odds change metric. This shows that heuristic linearization without Shapley approximation is suboptimal. However, the experiment does not show a clean gradient: SHAP, LIME, and New DeepLIFT are all within overlapping error bars, so it is not demonstrated that SHAP > New DeepLIFT or that Kernel SHAP (exact, expensive) > Deep SHAP (approximate, fast). The experiment validates that Shapley approximation helps relative to no approximation, but does not distinguish among different levels of approximation quality.

**What the experiment does not demonstrate:** The masking metric (change in log-odds when removing top-attributed pixels) measures a coarse form of causal relevance — do the highlighted pixels matter for the prediction? — rather than the *accuracy* or *fairness* of the attribution values themselves. Two methods could produce different attribution vectors but identify overlapping sets of top pixels, yielding similar masking performance despite different $\phi_i$ values. This metric cannot detect violations of local accuracy (do the attributions sum to the prediction?) or consistency (does a uniformly more important feature get a larger attribution?). The experiment uses a single model (the specific CNN from Shrikumar et al., 2017), a single input image (the "8" shown in Figure 5A), and a single class pair (8 vs. 3) for the detailed visualization; the quantitative results aggregate over 20 images but within the same model and dataset. This is a narrow slice of the space of deep network applications.

#### Genuine weaknesses in the experimental design:

1. **No direct comparison of Deep SHAP to Kernel SHAP.** The paper proposes Deep SHAP as a fast compositional approximation (Section 4.2) but never empirically validates that Deep SHAP's values approximate those from Kernel SHAP on the same model. Without this, the relationship between the two proposed methods is purely theoretical — users cannot know how much accuracy they sacrifice for the computational speedup.

2. **LIME comparison uses the original heuristics, not a controlled ablation.** Figure 3 shows LIME diverging from SHAP values, but does not isolate *which* of LIME's design choices (kernel, regularization, loss) causes the divergence. A reader wanting to fix LIME rather than abandon it does not learn what specifically to change. The paper's Theorem 2 provides the answer theoretically (use the Shapley kernel, no regularization), but this is not empirically validated by showing that a modified LIME with the Shapley kernel recovers SHAP values.

3. **User study has no quantitative agreement metric or statistical test.** The bar charts in Figure 4 visually suggest SHAP matches human judgments better than alternatives, but no correlation coefficient, mean absolute error, or hypothesis test is reported. The statistical significance of the observed differences is unknown.

4. **No experiment on a realistic high-stakes domain.** The paper's motivating examples (Section 1) invoke medical diagnosis, loan approvals, and scientific discovery — domains where interpretability failures have human consequences. All experiments use synthetic models, toy logical functions, or MNIST digit classification. The gap between the motivating stakes and the empirical validation is substantial.

5. **Computational cost of Kernel SHAP is not benchmarked against alternatives in wall-clock time.** Figure 3 uses "evaluations of the original model" as the cost metric, which is appropriate for model-agnostic methods where each evaluation dominates runtime. But for MNIST (Figure 5), Kernel SHAP uses 50,000 evaluations per image while DeepLIFT/Deep SHAP requires one backward pass. The paper does not report the actual runtime difference or memory requirements, which is the practical consideration for practitioners choosing between methods.

6. **The "New DeepLIFT" in Figure 5 is not Deep SHAP.** The paper introduces Deep SHAP (Equations 13–16) as a compositional method using exact component-level SHAP values, but the MNIST experiment uses "New DeepLIFT" from Shrikumar et al. (2017) — an improved but still heuristic version — and Kernel SHAP (model-agnostic). Deep SHAP itself is never empirically evaluated. This means the paper's primary proposed method for deep networks remains experimentally unvalidated.

7. **No comparison to a simple baseline like gradient×input.** Gradient×input (gradient of the output with respect to the input, multiplied element-wise by the input) is a standard baseline for neural network attribution that also produces additive explanations. The paper does not compare SHAP against this or other gradient-based methods, making it unclear whether the theoretical advantages of SHAP translate to practical improvements over simpler alternatives for deep networks.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost Is Unaccounted For in the Headline Efficiency Numbers

**The assumption or constraint.** The entire compute-optimal framework depends on estimating the difficulty of each prompt *before* deciding how to allocate the test-time compute budget. The paper's method for doing so — generating 2048 samples per question and averaging the PRM's final-answer scores across those samples — is extremely expensive. The paper explicitly acknowledges this in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

**The consequence.** The reported ~4× efficiency gains over best-of-N are computed *after* difficulty is known, without amortizing the cost of learning it. With 2048 samples per question for difficulty estimation, the total cost per question is: (difficulty estimation samples) + (strategy execution samples). For the scenarios where the compute-optimal policy uses small budgets (e.g., 16 generations matching best-of-64 performance), the difficulty estimation cost (2048 generations) swamps the strategy execution cost by two orders of magnitude. In a realistic deployment, the total cost would be dominated by difficulty estimation, and the ~4× figure would not be realized unless difficulty could be estimated much more cheaply — for example, by training a dedicated difficulty predictor or by amortizing the 2048 samples over many questions with similar characteristics.

**What evidence exists in the paper.** No experiment measures the total cost including difficulty estimation. The oracle bin experiments (Figures 4 and 8) use pre-computed difficulty bins and apply the compute-optimal policy without counting the samples used to construct those bins. The predicted bin experiments (same figures) use PRM-based difficulty estimation but similarly do not add the 2048 samples to the reported generation budget. The curves for oracle and predicted difficulty largely overlap, which is encouraging for the viability of PRM-based difficulty estimation, but neither curve reflects the true deployment cost. The paper provides no ablation studying how accuracy degrades if difficulty is estimated from fewer than 2048 samples.

**Mitigation status.** The paper acknowledges this as a limitation and frames it as an exploration-exploitation tradeoff for future work (Section 3.2, Section 8). No practical solution is provided. The paper suggests that future work could train models to directly predict difficulty from the question text, which would eliminate the per-question sampling cost entirely, but this is left as an unimplemented direction.

---

### Hard Problems Remain Essentially Unsolved — Test-Time Compute Cannot Create Capability

**The constraint.** Across all methods — PRM search, iterative revisions, and their compute-optimal combinations — the hardest questions (difficulty bin 5, roughly the bottom quintile of the MATH test set by base model pass@1) show near-zero improvement regardless of how much test-time compute is allocated. The base model's pass@1 on these problems is near zero, meaning there are essentially no correct solutions in the proposal distribution to find via search or to refine via revision.

**The consequence.** Test-time compute amplifies existing capability but cannot create it from nothing. If the base model fundamentally cannot solve a problem class — cannot produce a correct solution even occasionally — then no amount of search or revision will help. This means the SHAP framework offers no path forward for genuinely novel or out-of-distribution reasoning that exceeds the base model's training distribution. For these problems, pretraining remains the only viable path, as the FLOPs-matched comparison confirms: on hard problems at high inference-to-pretraining ratios, pretraining is strongly preferred over test-time compute (e.g., -52.9% relative disadvantage for PRM search on hard questions when R >> 1; Figure 9, bar chart in Figure 1). The paper is candid about this in Section 7:

> "Test-time compute provides essentially zero benefit regardless of budget, meaning that some capabilities can only be acquired through pretraining, not recovered at inference time."

**What evidence exists in the paper.** The difficulty-bin analyses consistently show bin 5 accuracy hovering at 1–3% for all methods and all budgets. In Figure 3 (right), bin 5 accuracy is near-zero for both best-of-N weighted and beam search across all budget levels (4 to 256 generations). In Figure 7 (right), bin 5 shows roughly 2–3% accuracy regardless of the sequential-to-parallel ratio at 128 generations. In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line is essentially flat near 0–5% accuracy, while the ~14× larger model's performance (stars) is consistently above this line by a substantial margin across all R values.

**Mitigation status.** The paper explicitly acknowledges this boundary condition and does not claim otherwise. The limitation is inherent to the approach rather than something that can be mitigated by a better algorithm — if the proposal distribution contains no correct answers, no selection mechanism can find one. Future work could potentially address this by combining test-time compute with retrieval-augmented generation (bringing in external knowledge at inference time) or by fine-tuning the base model on synthetic correct solutions from stronger models, but neither is explored in this paper.

---

### The ~14× Larger Model Baseline Is Not Compute-Optimal and Uses No Test-Time Compute of Its Own

**The assumption.** The FLOPs-matched comparison in Section 7 scales model parameters while holding training data fixed, following the LLaMA paradigm (Touvron et al., 2023) rather than the Chinchilla-optimal approach of scaling both data and parameters equally. The paper states:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

Additionally, the ~14× larger model is evaluated using only greedy decoding — no majority voting, no best-of-N, no search of any kind.

**The consequence.** Both design choices make the pretraining baseline *weaker than it needs to be*. A Chinchilla-optimal model trained with ~14× more total FLOPs (scaling both data and parameters appropriately) would likely outperform a parameter-only-scaled model, potentially reducing or reversing the test-time compute advantages reported in Figure 9. Giving the larger model even a modest test-time compute budget (e.g., best-of-8, which costs only 8× more inference per query) would create a much stronger baseline. The headline finding that test-time compute can substitute for a ~14× larger model is therefore qualified: it substitutes for a model that is larger in parameters but may not be optimally trained, and that is not using any test-time compute strategies itself. A fairer comparison would match total FLOPs while allowing both the smaller and larger model to use their respective optimal test-time strategies for a given inference budget fraction.

**What evidence exists in the paper.** The FLOPs-matched experiments in Figure 9 and the bar charts in Figure 1 report the comparison as described. The paper is transparent about the parameter-only scaling choice, acknowledging the departure from compute-optimal pretraining. No experiment tests the comparison against a Chinchilla-optimal larger model or against a larger model with any amount of test-time compute augmentation.

**Mitigation status.** The paper acknowledges the limitation in Section 7 ("we leave the analysis of compute-optimal scaling of pretraining compute... to future work") but does not provide even a sensitivity analysis. A partial mitigation would be to plot how the test-time compute advantage changes as a function of the larger model's test-time compute budget, showing at what point the advantage disappears.

---

### The Revision Model Has a ~38% Correct-to-Incorrect Reversion Rate

**The constraint.** The revision model was trained only on sequences where all in-context answers are incorrect (followed by a correct target). At test time, however, the model may produce a correct answer early in the revision chain and then "revise" it into an incorrect answer in a subsequent step. The paper reports (Section 6.1) that approximately 38% of correct answers get converted back to incorrect ones using a naive approach. This is a direct consequence of the training data construction: the model has no signal for what to do when the current answer is already correct, because such trajectories never appear in the training data.

**The consequence.** Sequential revision chains are inherently unstable. Even when the model finds the correct answer, it may discard it in favor of an incorrect one in the next revision step. The paper mitigates this with a selection mechanism (majority voting or verifier-based selection across the entire chain, picking the best answer from any point in the chain rather than always taking the last revision), but this is a post-hoc patch rather than a solution to the underlying problem. The reversion rate means that longer chains do not monotonically improve performance — the pass@1 trajectory in Figure 6 (left) shows gradual improvement but also fluctuations, and the fully-sequential curves in Figure 7 show diminishing and sometimes negative returns from additional sequential steps.

**What evidence exists in the paper.** The 38% figure is reported in Section 6.1. The performance of the ReST^EM-trained revision model (Appendix K, Figure 16) provides additional evidence of fragility: attempting to optimize the revision model with RL-style training caused performance to degrade substantially with sequential revisions. At 256 generations, fully sequential performance with the ReST^EM model drops to approximately 33.5% compared to roughly 38.5% at the optimal intermediate ratio, demonstrating that the revision approach is sensitive to training methodology in ways that are not fully understood.

**Mitigation status.** The paper addresses the reversion problem partially via (1) within-chain selection (majority voting or verifier-based selection across the entire revision chain rather than taking only the final revision), and (2) the compute-optimal policy, which often selects intermediate sequential-to-parallel ratios rather than fully sequential operation. However, these are mitigations that work around the problem rather than fixing it. A more principled solution — such as training the revision model to recognize when no revision is needed, or including correct-to-correct trajectories in the training data — is not explored. Section 8 does not list this as a specific direction for future work.

---

### Single Benchmark, Single Model Family — Generality Is Unproven

**The constraint.** All experiments use the MATH benchmark (500 test questions) with PaLM 2-S* as the base model. The paper states they "believe this model is representative of the capabilities of many contemporary LLMs" (Section 4), but this claim is unverified. The MATH benchmark consists exclusively of competition-level math problems requiring symbolic multi-step reasoning with exact ground-truth answers.

**The consequence.** Several aspects of the findings could be model-specific or domain-specific. The PRM's quality and over-optimization behavior depend on PaLM 2-S*'s output distribution — a model with different calibration properties, different error patterns, or a different pretraining corpus might exhibit different difficulty-dependent scaling curves. The revision model's ability to learn from incorrect in-context examples depends on the base model's in-context learning and self-correction capabilities, which vary substantially across model families and scales. The MATH benchmark's requirement for exact answer matching enables clean difficulty estimation (via pass@1) and PRM training (via Monte Carlo rollout correctness), but many important real-world applications — open-ended generation, dialogue, creative writing, complex planning — lack such clean correctness signals. Extending the compute-optimal framework to tasks where correctness is ambiguous or multi-dimensional would require fundamentally different verifier training and difficulty estimation approaches.

**What evidence exists in the paper.** The paper provides no experiments on other benchmarks, other model families, or other task domains. The reliance on MATH is acknowledged in Section 4 as a deliberate choice (test-time compute is expected to help most when the model already possesses the necessary knowledge and the challenge is drawing complex inferences), but the absence of even a second benchmark means the generality of the findings is unknown. The test set of 500 questions, split into five difficulty quintiles of ~100 each, then further split by two-fold cross-validation, means the compute-optimal policy is selected based on ~50 questions per fold per bin — a relatively small sample that could introduce variance in the computed-optimal policy. Confidence intervals on the compute-optimal scaling curves are not reported.

**Mitigation status.** The paper does not attempt to address this. The scope limitation is inherent to the empirical design and is not presented as a problem to be solved in future work. Section 8 suggests extending the framework to other domains ("code generation... logical reasoning... scientific QA") as a direction for future research, acknowledging the gap without filling it.

---

### No Accounting for Latency or Wall-Clock Time — Sequential Strategies Are Serially Bottlenecked

**The constraint.** The paper measures test-time compute in "generations" (number of complete solutions sampled), a reasonable proxy for total FLOPs but one that ignores latency. Sequential revisions are inherently serial — each revision depends on the previous one — while parallel best-of-N can be executed simultaneously with sufficient hardware parallelism. A strategy that allocates 128 generations as 64 sequential × 2 parallel (the compute-optimal choice for medium-difficulty problems in Figure 7) takes roughly 64× longer wall-clock time than one that runs 128 parallel samples simultaneously, assuming sufficient hardware to parallelize fully.

**The consequence.** For latency-sensitive applications — interactive assistants, real-time decision-making systems, user-facing chatbots — the sequential-heavy strategies favored by the compute-optimal policy on certain difficulty levels may be impractical regardless of their FLOPs-efficiency advantages. The paper does not discuss this tradeoff, and the compute-optimal policy does not include latency as a constraint or objective. A practitioner deploying these methods would need to independently evaluate whether the accuracy gains from sequential revisions justify the increased response time.

**What evidence exists in the paper.** None. The paper does not measure wall-clock time, latency, or throughput for any of the strategies. The generation budget metric inherently treats all generations as having equal cost, abstracting away the distinction between parallel and sequential execution.

**Mitigation status.** Not addressed at all. The paper does not mention latency as a consideration. This limitation is partially mitigated by the observation that the compute-optimal policy on easy problems (which are presumably common in many deployment scenarios) often selects low-budget strategies (e.g., 4–16 generations in Figure 4), which would have relatively low latency regardless of sequential vs. parallel allocation. However, the hardest problems that require mixed sequential-parallel allocation at high budgets (e.g., 128+ generations) would be exactly the ones where latency is most problematic, since users waiting for an answer to a hard question experience the longest delays.
