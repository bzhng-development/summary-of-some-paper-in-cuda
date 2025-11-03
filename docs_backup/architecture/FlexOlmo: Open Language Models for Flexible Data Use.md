# FlexOlmo: Open Language Models for Flexible Data Use

**ArXiv:** [2507.07024](https://arxiv.org/abs/2507.07024)
**Authors:** Weijia Shi, Akshita Bhagia, Kevin Farhat, Niklas Muennighoff, Pete Walsh, Jacob Morrison, Dustin Schwenk, Shayne Longpre, Jake Poznanski, Allyson Ettinger, Daogao Liu, Margaret Li, Dirk Groeneveld, Mike Lewis, Wen‑tau Yih, Luca Soldaini, Kyle Lo, Noah A. Smith, Luke Zettlemoyer, Pang Wei Koh, Hannaneh Hajishirzi, Ali Farhadi, Sewon Min
**Institutions:** Allen Institute for AI (AI2) and collaborators

## 🎯 Pitch

FLEXOLMO pioneers a novel modular approach to language model training that empowers data owners to independently train and manage model components without sharing data, utilizing a shared public model anchor and domain-specific routing to integrate them dynamically. This innovation addresses critical privacy and compliance challenges, offering significant accuracy gains while providing unprecedented control over data usage, making it pivotal for secure and adaptive AI deployments in regulated industries.

---

## 1. Executive Summary (2-3 sentences)
FLEXOLMO is a new way to build and use language models that lets multiple data owners train specialized model modules on their own closed data without sharing that data, then plug those modules together at inference time with fine-grained opt-in/opt-out control (Figure 1; §3). It achieves this by training mixture-of-experts (MoE) components independently with a shared “public” anchor and a domain-informed router, delivering strong accuracy gains and flexible data usage—e.g., an average 10.1% relative improvement over the best prior merging baseline (BTM) across 31 tasks (Table 2), while enabling strict exclusion of experts (Figure 4).

## 2. Context and Motivation
- The problem (what’s missing today)
  - Standard large language model pretraining centralizes data, forcing one-time inclusion/exclusion decisions and making it hard to later remove the influence of specific data (§1; §2.1). This blocks the use of valuable closed datasets (e.g., medical, finance) and complicates compliance with licensing, consent, and role-based access constraints.
  - Existing “machine unlearning” is limited for LLMs, offering weak guarantees or impractical costs (§1; refs [1–3]).
- Why this matters
  - Many real-world datasets cannot be shared due to privacy, regulation, IP, or business needs (HIPAA, GDPR, data sovereignty; §2.1). Organizations need:
    - Training without data pooling, and
    - Inference-time control to include/exclude data sources per use case (privileged access, license restrictions, content safety; §2.1).
- Prior approaches and their gaps
  - Federated learning avoids data centralization but suffers from high synchronization cost, performance degradation, and privacy attack surface due to communication (§2.2; refs [4–7, 37–38]).
  - Model merging techniques (e.g., model soup, weighted ensembling/BTM) can combine specialized models but lack learned connections among components, limiting expressivity (§2.2).
  - Prior MoE upcycling approaches (e.g., BTX, DEMix-style) require joint training on the union of datasets after merging, which breaks the “no joint data access” constraint (§2.2).
- Positioning of this work
  - FLEXOLMO creates a modular MoE system where each expert is trained independently on closed data (no joint access) but later composes cleanly through a router designed to be assembled by concatenation (§3.3.2). It combines benefits of specialization with data-governance guarantees at inference time (Figure 1), and shows superior accuracy over ensembling/merging baselines (Tables 1–2).

## 3. Technical Approach
At a high level, FLEXOLMO replaces each transformer block’s feedforward network (FFN) with a set of independently trained FFN “experts” and a `router` that chooses which experts to activate per token (§3.2). The key challenge is enabling experts, trained in isolation on disjoint data, to “work together” without joint training.

Step-by-step:

1) Architecture: sparse Mixture-of-Experts (MoE) with independent experts (§3.2)
- What is an MoE? An architecture where multiple expert sub-networks (here, FFNs) exist per layer, and a small subset (top-k) is chosen per token by a `router`, improving capacity without always running all experts.
- Notation:
  - Input token representation `x ∈ R^h`.
  - Experts `{M_pub, M_1, …, M_n}` where `M_pub` is a public-data expert; `M_i` comes from closed dataset `D_i`.
  - Router `r(x) = W_r x`, with `W_r ∈ R^{(n+1)×h}` producing a score for each expert; top-k experts get activated.
  - Output at a MoE layer:
    > y = Σ_{i ∈ Topk(r(x))} softmax(r(x))_i · M_i(x)  (Eq. in §3.2)

2) Training experts to coordinate without joint data (§3.3.1; Figure 1)
- Problem: If each expert `M_i` is continued-pretrained alone, their representations drift, and merging them later is brittle.
- Solution: Train each expert in a two-expert MoE “sandbox” with a frozen shared anchor:
  - Build a 2-expert MoE: the frozen public expert `M_pub` (and frozen attention layers) + a trainable expert `M_i`.
  - Train `M_i` only on `D_i`, holding `M_pub` and attention fixed. This “teaches” `M_i` to complement `M_pub` in a shared coordinate system, making cross-expert coordination feasible at merge time (§3.3.1).
  - During this step, learn router embeddings “pairwise” only between `[r_pub, r_i]` (details next).

3) Domain-informed router learned by concatenation (§3.3.2)
- Router as stacked embeddings:
  - `W_r` is built by stacking per-expert router embeddings: `[r_pub; r_1; …; r_n]`.
  - Each `r_i` is initialized from a “domain embedding”: average of document embeddings from `D_i` via an external embedder `E` (they use GRIT, a generative instruction-tuned embedder; §3.3.2; Table 3).
  - During a given expert’s local training, only `[r_pub, r_i]` is present; `r_pub` remains the same across all experts, ensuring all pairwise training shares a common anchor (§D.2).
- Merging by concatenation:
  - After every expert is trained independently, the final router is literally created by concatenating all learned rows into `W_r`. No joint finetuning is required (§3.3.2).
- Why this works:
  - The frozen `r_pub` and public expert act as a shared reference frame, so all `r_i` live in the same “score space” (§D.2). At inference, a global `argmax` or top-k works as if trained jointly.

4) A small but important calibration: a negative bias per expert (§3.3.2; §D)
- Challenge: Each expert was trained only against the public expert, not against other experts. At inference, experts must compete fairly with all others.
- Fix: Add a learned negative bias `b_i` for each `M_i` and select `M_i` only if `r_i·x + b_i > r_pub·x` (Eq. in §3.3.2; intuition in §D.2). This nudges decision boundaries so experts fire only when strongly appropriate, mitigating over-activation once many experts are present.

5) Optional router-only tuning on public proxy data (§3.3.3)
- Data owners may identify a small “proxy” subset `D̂_i ⊂ D_pub` that resembles their closed data (built via a binary classifier between `D_i` and `D_pub` to find similar public samples; §A.2).
- After merging, lightly finetune only the router embeddings on the union of all `D̂_i` and `D_pub` (no closed data are used in this step). This modestly improves routing quality (Tables 1–2, “no RT” vs “FLEXOLMO”).

6) Data pipeline and training scale (§4.1, §4.4)
- Public base model `M_pub`: a 7B-parameter dense transformer following OLMo-2 with 32 layers, hidden size 4096, trained for 1T public tokens (§4.4).
- Closed data are simulated via seven domain sets: News, Creative Writing, Code, Academic, Educational Text, Math, Reddit (FLEXMIX; §4.1; Figure 5).
- Each expert is continued-pretrained for 50B tokens on its domain (total 400B across 8 experts; §4.4). Final 8-expert FLEXOLMO has 37B total parameters with 20B active at inference (top-4 experts) (§4.4; Figure 3).

Analogy for intuition:
- Think of `M_pub` as a shared lingua franca and each `M_i` as a specialist with its own dialect. Specialists learn to converse with the lingua franca during training; later, the router selects which specialists to consult per token. Concatenation works because everyone learned to speak (score) in the lingua franca’s coordinates.

## 4. Key Insights and Innovations
1) Independent expert training anchored by a frozen public model (§3.3.1; Figure 1)
- What’s new: Experts are not trained jointly; each is trained with a frozen `M_pub` and shared attention, so all specialists remain compatible.
- Why it matters: Avoids any joint access to closed datasets while still enabling unified behavior at inference. Ablation shows removing this “learn to coordinate” step drops the average from 46.7 to 38.8 in the 4-expert setting (Table 1, “- no training to coordinate”).

2) Router-by-concatenation with domain-informed initialization (§3.3.2)
- What’s new: Router weights are per-expert embeddings, initialized from domain text embeddings (GRIT) and learned pairwise; the final router is assembled by stacking rows—no joint training needed.
- Why it matters: This is the crux enabling “plug-in” specialists and truly flexible opt-in/opt-out at inference (Figure 1). Using GRIT initializations improves the 4-expert average from 43.5 to 46.7 (Table 3).

3) Negative bias to bridge pairwise and multiclass routing (§3.3.2; §D)
- What’s new: A simple bias term per expert corrects for the fact that experts were only trained to beat `M_pub`, not each other.
- Why it matters: It calibrates competition among many experts, improving robustness when merging. Removing the bias drops the 4-expert average from 46.7 to 45.8 (Table 1, “- no bias”).

4) Flexible, enforceable data opt-in/opt-out at inference (Figure 1; §5.2)
- What’s new: Excluding a dataset reduces to removing its expert module and router row—no retraining. Figure 4 demonstrates “opting out” the News expert significantly reduces News generation scores while leaving unrelated tasks largely unchanged.
- Why it matters: This directly serves licensing, consent, and role-based access needs that motivate the work (§2.1).

These are fundamental innovations (new training+router design for modular MoE under data constraints), not just incremental tuning.

## 5. Experimental Analysis
Evaluation setup (§4.2–§4.4)
- Data: FLEXMIX includes 1 public mix + 7 domain sets designed to approximate closed datasets (News, Creative Writing, Code, Academic, Educational, Math, Reddit; §4.1; Figure 5).
- Benchmarks and metrics (31 tasks across 10 categories; §4.2; Table 5):
  - General-purpose: MC9 (9 MC tasks), GEN5 (5 QA/NQ tasks), MMLU, MMLU-Pro, AGIEval, BBH.
  - Domain-specific: Math2 (GSM8K, MATH), Code4 (MBPP, MBPP+, HumanEval, HumanEval+), SciRIFF5 (5 scientific QA tasks), News Generation, Poem Generation.
- Baselines (§4.3): Prompt-based routing (Llama 3.1 and OLMo routers), Model Soup (avg/weighted), BTM (probability ensembling), BTX (MoE upcycling with public-only post-merge training), Unrestricted MoE (upper bound with joint data access; both FLOPs-controlled and data-controlled).
- Training details (§4.4): `M_pub` 7B dense on 1T public tokens; each expert 50B domain tokens; optional router tuning on 5B public proxy tokens. Final 8-expert model: 37B total params; 20B active (top-4).

Main quantitative results, with numbers and comparisons

Small-scale (4 experts: Public, Math, Educational, Code; 24 tasks; Table 1)
- Average performance:
  - > Previous public model: 36.9
  - > Best baseline (BTM): 43.4
  - > FLEXOLMO (no router tuning): 46.7
  - > FLEXOLMO (with router tuning): 47.8
- Key takeaways:
  - FLEXOLMO beats all merging baselines; relative gain over BTM ≈ 10.1% at 8-expert scale (see below), and 10.1% claim overall (§5.1).
  - It also exceeds the Unrestricted MoE under the FLOPs-controlled setting (1× FLOPs, 0.5× data): 47.8 vs 46.3 (Table 1), supporting the claim:
    > “FLEXOLMO … surpasses the standard MoE trained without data restrictions using the same training FLOPs.” (Abstract; Table 1)
  - Against data-controlled Unrestricted MoE (2× FLOPs), FLEXOLMO trails (47.8 vs 51.5).

Large-scale (8 experts; 31 tasks; Table 2)
- Average performance:
  - > Previous public model: 42.4
  - > BTM (top-2): 47.6
  - > FLEXOLMO (no router tuning): 51.3
  - > FLEXOLMO (with router tuning): 52.4
- Category gains vs public baseline:
  - MC9: 70.8 vs 68.7
  - MMLU: 60.4 vs 55.9; MMLU-Pro: 30.9 vs 26.2
  - BBH: 46.4 vs 35.7
  - Math2: 48.5 vs 8.2
  - NewsG: 80.7 vs 76.0; PoemG: 62.2 vs 47.8
  - SciRIFF5: 54.3 vs 48.1
  - Code4: 17.2 vs 1.1 (significant improvement, though BTM is higher here; see below)
- Relative improvement over BTM (top-2):
  > “FLEXOLMO … outperforms prior model merging methods by 10.1% on average” (Abstract; Table 2 shows 52.4 vs 47.6).
- Where baselines shine or struggle:
  - BTM is the strongest baseline overall, but FLEXOLMO still adds ≈4.8 points average (52.4 vs 47.6).
  - For Code4, in both 4-expert and 8-expert settings, BTM slightly outperforms FLEXOLMO (e.g., 24.0 vs 17.2 in Table 2). FLEXOLMO still dramatically improves over the public baseline (+16.1 points), but code problems are a case where ensembling gains more.

Ablations and router choices (Table 1; Table 3)
- Removing any core piece hurts:
  - No coordinate training: 38.8 (–9.0)
  - No bias: 45.8 (–1.0)
  - Random router init, no bias: 44.4 (–2.3)
- Router initialization with GRIT vs using public model embeddings:
  - > “GRIT embedder consistently outperforms public model embeddings” (Table 3). Average +3.2 points (46.7 vs 43.5).

Behavior analysis (Figure 2; Figure 3; Figure 4)
- Routing patterns: Tokens from different domains predominantly activate their matching expert, with frequent public expert activation as a complement. Different layers activate different expert combinations (Figure 2), showcasing token- and layer-wise specialization.
- Active experts vs performance: Gains plateau after activating 4 experts (Figure 3), motivating the design choice of top-4 for efficiency (20B active of 37B total; §4.4).
- Opt-out behavior: Removing the News expert specifically lowers News Generation performance with minimal effect on other tasks (Figure 4), validating “data-flexible inference.”

Data extraction risk (§5.3)
- Attack setup: Sample 10k math documents; prompt with 32-token prefixes; generate 256 tokens; count as extracted if normalized Levenshtein ≥0.9; 10 samples per prefix.
- Results:
  - > Public model (no math data): 0.1%
  - > Dense math expert: 1.6%
  - > FLEXOLMO with math expert: 0.7%
- Interpretation: Extraction is nonzero once weights see the data. The work recommends per-expert differential privacy if formal guarantees are required (§5.3).

Scaling to a stronger base (§5.4; Table 4)
- Starting from OLMo-2 7B’s 4T-token checkpoint, the FLEXOLMO recipe with two additional experts (math, code; 50B tokens each) outperforms the final OLMo-2 7B on the same total training FLOPs:
  - > Average: 52.8 (FLEXOLMO) vs 49.8 (OLMo-2 7B) (Table 4)
  - Inference cost rises (2.5× FLOPs) due to activating 3 experts; see trade-offs below.

Do the experiments support the claims?
- Yes, on three axes:
  - Performance: Consistent average gains over strong baselines across two scales (Tables 1–2), especially on domains with specialized experts (BBH, Math2, SciRIFF5, NewsG, PoemG).
  - Modularity: Router concatenation works in practice; ablations show each design element matters (Table 1; Table 3).
  - Data control: Opt-out behaves as intended (Figure 4).
- Nuances:
  - Code4 is a notable outlier where BTM ensembling edges out FLEXOLMO, despite large gains vs the public baseline (Tables 1–2).

Selected paper claims with citations
- > “FLEXOLMO improves upon the public model by 41%” (Abstract; reiterated in §5.1). Table 2 shows +10 points on the 8-expert overall average (42.4 → 52.4; +23.6% relative), suggesting the 41% figure refers to a different aggregation; nonetheless, broad improvements are clear across benchmarks.
- > “Outperforms prior model merging methods by 10.1% on average” (Abstract; Table 2: 52.4 vs 47.6).
- > “Surpasses the standard MoE trained without data restrictions using the same training FLOPs” (Abstract; Table 1: 47.8 vs 46.3).

## 6. Limitations and Trade-offs
Assumptions and scope
- Assumes each data owner can continue-pretrain a sizable expert (50B tokens in experiments; §4.4). Very small closed datasets may under-train experts.
- Only FFN submodules are specialized; attention remains shared and frozen during expert training (§3.3.1). Tasks needing attention-level specialization may benefit less.

Performance vs inference cost
- Flexibility comes with extra inference FLOPs: activating top-4 experts means ≈20B active params out of 37B total (§4.4; Figure 3). In the OLMo-2 extension, 3 active experts raise inference FLOPs to 2.5× of dense (Table 4).

Routing calibration and data heterogeneity
- Router learned via pairwise training plus bias is elegant but approximate. If domains heavily overlap, router confusion may increase. Optional router-only tuning on proxy public data helps but depends on the quality of proxy selection (§3.3.3; §A.2).

Privacy and guarantees
- While data never leave owners during training, sharing expert weights can leak a small fraction of memorized content; extraction observed at 0.7% on the math expert in FLEXOLMO (§5.3). Formal privacy requires per-expert differential privacy training, which is orthogonal but not built-in (§5.3).

“Complete removal” semantics
- Opting out by removing an expert guarantees that expert’s parameters no longer influence outputs (§3.1; §3.3.2). If proxy public data were used for router tuning, traces of closed-domain patterns could persist in router embeddings; however, the method allows dropping router rows too. The work does not provide formal unlearning proofs for any residual cross-module effects.

Code-domain caveat
- In both 4- and 8-expert settings, BTM slightly outperforms FLEXOLMO on Code4 (Tables 1–2). This suggests some problem classes may benefit more from ensembling than sparse expert routing.

## 7. Implications and Future Directions
How this changes the landscape
- Turns LMs into modular systems where capabilities can be added, updated, or removed without re-centralizing data or re-training the full model (Figure 1; §3). This is a new capability for data governance: per-deployment “data eligibility” encoded as which experts are present.

Practical applications
- Regulated domains (healthcare, finance, enterprise): Each department or org trains an expert locally; deployment includes only experts the user has rights to use (Figure 4).
- Content-sensitive deployments: Keep a toxicity expert active for lab analysis; remove it for consumer-facing apps (§2.1, “Model control”).
- Time-bound data rights: Remove an expert when a license expires, without model retraining.

Research directions
- Stronger privacy: Integrate per-expert differential privacy to add formal guarantees (§5.3).
- Router learning: Explore multiclass-consistent training without joint data (beyond pairwise + bias), better domain embedding initializations, and adaptive top-k per token/layer (§3.3.2; Figure 2–3).
- Expert marketplaces: Standardize expert formats and router rows so external teams can publish “plug-in” experts with clear licenses and usage constraints.
- Compression/distillation: Distill a selected subset of experts into a smaller dense model for low-latency deployments while preserving opt-out semantics.
- Beyond text: Extend modular, data-controlled MoE training to multimodal settings (e.g., code+vision) and retrieval-augmented pipelines.
- Robustness and safety: Study fairness, distribution shift, and adversarial routing when experts are independently produced by different parties.

In short, FLEXOLMO offers a concrete, empirically validated recipe for training and deploying LMs under real-world data restrictions, with a clean mechanism for composition and control. The results in Tables 1–2, Figures 2–4, and §5.4 show that modularity does not have to come at the cost of accuracy, especially when routing and anchoring are designed to make independently trained experts interoperable.
