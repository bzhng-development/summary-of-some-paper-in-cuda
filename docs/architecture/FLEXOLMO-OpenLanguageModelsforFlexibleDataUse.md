# FLEXOLMO: Open Language Models for Flexible Data Use

**ArXiv:** [2507.07024](https://arxiv.org/abs/2507.07024)

## 🎯 Pitch

FLEXOLMO introduces a new framework for training and merging language model 'experts' independently on closed datasets without requiring any data sharing or joint retraining. This breakthrough enables organizations to benefit from sensitive or regulated data while retaining strict ownership and opt-out controls at inference, all while achieving strong performance gains over public-only models and prior merging techniques. By empowering fine-grained data control without sacrificing accuracy, FLEXOLMO fundamentally expands the possibilities for collaborative AI development in domains with stringent data restrictions.

---

## 1. Executive Summary
FLEXOLMO introduces a way to train and use language models when data cannot be shared: different “experts” are trained locally on closed datasets and later combined—without any joint retraining—into a single mixture-of-experts model that can flexibly include or exclude those experts at inference. This matters because it enables organizations to benefit from their sensitive data while preserving control and offering strict opt-out guarantees, and it yields strong accuracy gains over both a public-only model and prior merging baselines (see §5 and Tables 1–2).

## 2. Context and Motivation
- The problem or gap
  - Standard pretraining centralizes data and offers little control over which data influences model behavior or how to remove it later (§1, §2.1). This blocks use of regulated or confidential datasets (e.g., in healthcare or finance), and it prevents per-user or per-application control over data usage at inference.
  - Two needs emerge (§2.1): (1) train without pooling data; (2) choose which data sources “count” at inference (opt-in/opt-out).
- Why it matters
  - Real-world impact: data sovereignty, regulatory compliance (HIPAA, GDPR), and evolving copyright/consent regimes require fine-grained control of data use (§2.1). Applications such as code assistants may need to reflect private repos only for authorized users.
  - Research significance: we need mechanisms that preserve modularity and provenance—allowing later removal of a dataset’s influence—without performance collapse.
- Prior approaches and limitations (§2.2)
  - Federated Learning: avoids data pooling but requires synchronous communication, suffers performance degradation, and is vulnerable to certain privacy attacks; adoption for LMs remains limited (§2.2).
  - Model merging (e.g., weight averaging “model soup,” output ensembling/BTM): can combine independently trained models, but lacks learned “connections” between modules, reducing expressivity when datasets are disjoint (§2.2).
  - Prior MoE upcycling (e.g., DEMix/BTX): typically requires joint training on the combined data after merging, which is incompatible with strict data separation (§2.2).
- How this work positions itself
  - FLEXOLMO builds an MoE where each expert is trained independently on its local data and later merged with a new domain-informed router—no joint data access or joint training is required (§3). It aims to preserve modularity and enable inference-time opt-in/opt-out while improving accuracy over public-only models and prior merging techniques (Tables 1–2).

## 3. Technical Approach
Key terms used here:
- `expert`: a small feed-forward network (FFN) inside each transformer block that specializes on a domain. In MoE, multiple experts exist and a `router` chooses which ones process each token.
- `router`: a lightweight module that scores experts for a token and activates the top-k experts.
- `router embedding`: a learned vector per expert; stacking these vectors forms the router’s weight matrix.
- `domain embedding`: an embedding summarizing a dataset/domain, used to initialize an expert’s router embedding.
- `opt-in/opt-out`: the ability to include or exclude an expert (and thus its data’s influence) at inference by adding/removing its router row.

Step-by-step design

1) Problem setup (§3.1)
- Start from a public dense model `Mpub` trained on public data `Dpub`.
- There are n closed datasets `D1…Dn`, each owned by a different party.
- Goal: produce one model `Mfinal` by composing `Mpub` with independently trained expert modules `M1…Mn` such that:
  - No party needs joint access to all data during training (each trains its own expert locally).
  - Removing expert `Mi` at inference guarantees removal of `Di`’s influence.

2) Architecture (§3.2)
- Use a standard transformer where the FFN in each block is replaced by an MoE module containing `n+1` experts: the public expert (`Mpub`) plus `n` closed-data experts (`M1…Mn`).
- Given token representation `x`, the router computes scores `r(x)=Wr x`, where `Wr` stacks one router embedding per expert. The module outputs a weighted sum of the top-k experts’ outputs.
  - Plain-language view: each token consults a few specialists; their weighted outputs are combined. This is done per token, per layer, so different layers/tokens can pick different experts.

3) Training experts to coordinate (§3.3.1, Fig. 1)
- Challenge: independently trained experts may not “fit together” at merge time.
- Solution: pairwise, anchored training with the public model.
  - For dataset `Di`, construct a temporary 2-expert MoE: a frozen copy of the public expert (`Mpub`) and a trainable expert (`Mi`). Attention layers are shared and frozen.
  - Train `Mi` on `Di` while learning a pair of router embeddings `[rpub, ri]` for “public vs. `i`” routing. `rpub` stays fixed across all trainings; `ri` is trained only on `Di`.
  - Intuition: freezing the public expert and its embedding creates a common coordinate system across all pairwise trainings (§D.2). Each `Mi` learns to complement, not replace, `Mpub`, improving later multi-expert compatibility.

4) Domain-informed router and merging-by-concatenation (§3.3.2)
- During each expert’s local training, the router embedding `ri` is initialized using a domain embedding:
  - Compute `ri = average_{dk∈Si} E(dk)`, where `E` is an off-the-shelf embedder (GRIT), and `Si` is a sample subset of `Di`. This initialization helps experts start from domain-aware positions (§3.3.2; Table 3 shows it matters).
- After all experts are trained, merge simply by stacking the router embeddings:
  - `Wr = [rpub; r1; …; rn]`. No joint retraining is needed. To opt-out an expert, remove its row.
- Add a negative bias `bi` to each expert during routing:
  - Select `Mi` only if `ri·x + bi > rpub·x` (§3.3.2). This compensates for the fact that each expert was only trained against the public model, not against other experts. The bias makes experts intervene only when they are clearly relevant. §D.2 explains how the bias nudges decision boundaries to be more selective, enabling better multi-class behavior when experts later compete.

5) Optional router tuning on public proxy data (§3.3.3)
- If data owners can identify public samples that resemble their closed data, perform a lightweight router-only finetune:
  - Build tiny proxy sets `D̂i ⊆ Dpub` by training a small classifier to distinguish `Di` vs. `Dpub`, then pick public samples most similar to `Di`.
  - After merging, finetune `r1…rn, rpub` on `D̂1…D̂n` and `Dpub`.
  - This uses only public data and tends to improve routing quality (Tables 1–2, “no RT” vs “FLEXOLMO”).

6) Implementation snapshot (§4.4)
- Public model: a 7B dense OLMo 2–style model trained on 1T public tokens.
- Each expert is continued-pretrained for 50B tokens on its local data (8 experts total → 400B tokens).
- Final 8-expert FLEXOLMO has 37B total parameters with 20B active at inference (top-4 experts activated). This sparsity yields efficiency versus a fully dense 37B model.

Analogy
- Think of FLEXOLMO as a panel of specialists who all trained alongside the same generalist (“public” expert). The meeting’s chair (router) uses a short bio (router embedding) for each specialist to decide—per word and per layer—who speaks. Anyone can be uninvited (opt-out) by removing their chair card, without retraining the rest.

## 4. Key Insights and Innovations
- Independent expert training anchored to a frozen public expert (§3.3.1)
  - Novelty: Experts are trained locally and asynchronously but still learn to cooperate because the public expert/embedding stays fixed as a shared anchor. This avoids the need for costly, privacy-sensitive joint training, yet preserves compatibility at merge-time.
  - Significance: Enables modular, distributed training on closed datasets that never leave their owners while improving downstream performance (Tables 1–2).

- Domain-informed router assembled by concatenation (§3.3.2)
  - Novelty: The router is a simple stack of per-expert embeddings, each initialized from domain embeddings and trained locally against the public expert. No post-merge joint training is required.
  - Significance: Merging reduces to concatenation; opt-in/opt-out reduces to adding/removing a row—practical, auditable control with strong provenance guarantees.

- Negative bias for selective expert activation (§3.3.2; §D.2)
  - Novelty: A per-expert negative bias corrects the fact that experts were only trained pairwise against the public expert, not against each other. The bias forces experts to intervene only on clearly in-domain tokens.
  - Significance: Improves multi-class routing after merging and is empirically important (ablation “- no bias” in Table 1 drops average accuracy by ~1 point).

- Router-only tuning using public proxy samples (§3.3.3)
  - Novelty: A tiny, public-only post-merge step refines the router without touching expert weights or requiring closed data.
  - Significance: Consistently helps average accuracy (compare “no RT” vs “FLEXOLMO” in Tables 1–2), preserving data separation.

Fundamental vs. incremental
- Fundamental: the modular training-and-merging recipe that enforces compatibility via a shared anchor and domain-informed concatenative routing; inference-time opt-out with no retraining.
- Incremental but important: the negative bias and proxy-based router tuning that stabilize and polish routing quality.

## 5. Experimental Analysis
- Evaluation setup (§4.2; Table 5)
  - Data: FLEXMIX contains one large Public Mix (Common Crawl–derived) plus seven domain-specific sets: News, Creative Writing, Code, Academic, Educational Text, Math, Reddit (§4.1; data sizes in Fig. 5/Appendix B).
  - Tasks: 31 benchmarks across 10 categories: general QA and reasoning (MC9, GEN5, MMLU, MMLU-Pro, AGIEval, BBH), and domain evaluations (Math2, Code4, SciRIFF5), plus two generation tasks judged by an LM (NewsG, PoemG) (§4.2; Table 5).
  - Baselines (§4.3): prompt-based routing (two LLM routers), model soup (avg/weighted), BTM (output ensembling), BTX (MoE upcycling with limited public-only post-merge training), and an “Unrestricted MoE” trained on all data as an upper-bound reference.
  - Training regime (§4.4): Public 7B model on 1T tokens; each expert continued for 50B tokens; final 8-expert model with 20B active parameters at inference (top-4 experts).

- Main quantitative results
  - 4-expert setting (public + math + code + educational) on 24 tasks (Table 1):
    - Average accuracy: 
      - Public dense model: 36.9
      - Best prior baseline (BTM): 43.4
      - FLEXOLMO: 47.8
    - Big domain gains:
      - BBH: 35.6 → 47.1
      - Math2: 8.1 → 50.7
      - Code4: 1.0 → 17.3
    - Upper bound comparison:
      - Unrestricted MoE (1× FLOPs, 0.5× data): 46.3 (below FLEXOLMO’s 47.8)
      - Unrestricted MoE (2× FLOPs, full data): 51.5 (above 47.8)
    - Takeaway: FLEXOLMO beats compute-matched unrestricted MoE and approaches the data-matched upper bound without joint training.
  - 8-expert full setting on 31 tasks (Table 2):
    - Average accuracy:
      - Public dense: 42.4
      - BTM (top-2): 47.6
      - FLEXOLMO (no RT): 51.3
      - FLEXOLMO (with RT): 52.4
    - Notable category gains over public:
      - NewsG: 76.0 → 80.7
      - PoemG: 47.8 → 62.2
      - SciRIFF5: 48.1 → 54.3
      - Math2: 8.2 → 48.5
      - Code4: 1.1 → 17.2
    - Relative to best prior baseline:
      > “FLEXOLMO outperforms all prior model merging methods, beating the best baseline BTM by 10.1% relative on average.” (§5.1; compare Table 1: 47.8 vs 43.4; and Table 2: 52.4 vs 47.6)

- Ablations and design choices
  - Coordinated training matters (Table 1):
    - Removing it (“- no training to coordinate”) drops average to 38.8 (vs 47.8).
  - Router initialization matters (Table 3):
    - GRIT domain embedding init yields 46.7 vs 43.5 with “Public” init.
  - Negative bias matters (Table 1):
    - “- no bias” reduces average to 45.8; “- no domain embedding init, no bias” to 44.4.
  - Optional router tuning (RT) helps overall (Tables 1–2):
    - 4-expert average: 46.7 (no RT) → 47.8; 8-expert average: 51.3 (no RT) → 52.4.
    - Trade-off: in the 8-expert setting, Code4 slightly decreases with RT (18.6 → 17.2), illustrating routing calibration can shift per-domain performance.

- Behavioral analyses
  - Routing patterns (Fig. 2): math inputs activate the math expert more, code activates code, etc., while the public expert remains frequently active—evidence the experts complement the public one across layers.
  - Number of active experts (Fig. 3): MMLU improves as more experts are active and plateaus around 4, supporting the “top-4” sparsity choice.
  - Opt-out (Fig. 4): removing the news expert reduces NewsG substantially while leaving unrelated tasks nearly unchanged—a concrete demonstration of inference-time data control.

- Data extraction risk (§5.3)
  - Extraction procedure (Carlini-style) on math data:
    - Public model (no math): 0.1% extraction.
    - Dense math expert: 1.6%.
    - FLEXOLMO with math expert: 0.7%.
  - Interpretation:
    > It is difficult to extract a substantial fraction of training data, but the risk is nonzero when any weights were trained on that data (§5.3). The paper recommends optional differentially private training for experts when needed.

- Scaling test with a stronger public model (Table 4; §5.4)
  - Start from OLMo-2 7B pre-anneal (4T tokens), add only two experts (math, code).
  - With equal training FLOPs, FLEXOLMO reaches 52.8 average vs OLMo-2 7B’s 49.8, with large math/code gains.
  - Inference cost: ~2.5× FLOPs vs a dense 7B (due to activating 3 experts), a known MoE trade-off.

- Overall assessment
  - The experiments support the core claims: (1) modular training without data pooling; (2) effective merging without joint training; (3) competitive or superior performance vs. strong merging baselines; (4) practical opt-out at inference with minimal collateral impact (Fig. 4).
  - The upper-bound gap (vs. unrestricted MoE with 2× FLOPs) is modest (Table 1), and design ablations clarify what drives performance.

## 6. Limitations and Trade-offs
- Assumptions and prerequisites
  - Requires a shared public base model and frozen public expert/embedding during each expert’s local training (§3.3.1). If organizations cannot agree on this anchor, coordination may weaken.
  - Router is linear (`Wr x`) with per-expert biases—simple and auditable, but less expressive than jointly trained nonlinear routers.
- Scenarios not fully addressed
  - Cross-expert interference when two closed datasets are highly overlapping is only implicitly handled via the negative bias; there is no explicit joint deconfliction step at merge time.
  - Closed datasets in experiments are “real or simulated” approximations (§4.1); behavior on highly idiosyncratic or noisy private corpora remains to be demonstrated.
- Computational and scaling considerations
  - Inference cost grows with the number of active experts (top-k). While results plateau at 4 experts (Fig. 3), large fleets of experts still raise routing and memory overhead.
  - Training requires continued pretraining (50B tokens per expert here), which is substantial even if cheaper than re-training a large joint model (§4.4).
- Privacy and governance
  - Data extraction is nonzero when sharing expert weights (0.7% on math for FLEXOLMO; §5.3). Differential privacy is suggested but not evaluated end-to-end in this work.
  - Optional router tuning uses proxy samples selected by a classifier; the quality of these proxies depends on classifier accuracy and may introduce subtle distribution shifts (§3.3.3; A.2).
- Performance trade-offs
  - Router tuning can slightly help average but harm a particular domain (e.g., Code4 in Table 2).
  - FLEXOLMO does not surpass the data-rich unrestricted MoE when that model uses more FLOPs/data (Table 1, bottom).

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes a practical recipe for collaborative, privacy-aware LM improvement: experts can be trained locally, shared as weights, merged by concatenation, and opted-out at runtime—no joint data access or retraining needed (§3, Fig. 1).
  - Introduces a clear “control plane” for data governance: each expert corresponds to a dataset; including/excluding it precisely controls that dataset’s influence (Fig. 4).
- Follow-up research enabled or suggested
  - Stronger routers without joint training: e.g., richer per-expert metadata, learned token-level domain cues, or meta-learning to improve generalization across many experts.
  - Formal privacy: end-to-end differentially private expert training and auditing pipelines; certified opt-out/unlearning guarantees building on recent work cited in §2.
  - Automated expert discovery: unsupervised domain discovery and expert creation to avoid manual domain curation (§2.2 references DEMix/BTX-like lines).
  - Economics and orchestration: protocols for multi-organization contribution, provenance tracking, licensing-aware gating, and compensation models.
  - Beyond text: extending the approach to multimodal experts (e.g., code+images, scientific figures), and to multilingual or cross-domain transfer.
- Practical applications
  - Regulated industries (healthcare, finance, government) that need to keep data local but still leverage LMs for analysis, decision support, or retrieval-augmented generation.
  - Enterprise assistants with per-user access control: include an organization’s private “code” or “docs” expert only for authorized users; remove it instantly when permissions change.
  - Research consortia: organizations contribute domain experts (e.g., materials science, clinical notes) to a common model without sharing raw data, accelerating scientific discovery.

In short, FLEXOLMO provides a principled, empirically validated path to modular LM training and data-flexible inference. Its anchor-based expert training and domain-informed router concatenation are the key mechanisms that make “train locally, merge globally, control at runtime” both feasible and effective (§3; Tables 1–2; Figs. 2–4).
