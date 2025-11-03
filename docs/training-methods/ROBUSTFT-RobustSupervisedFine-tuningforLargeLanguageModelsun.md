# ROBUSTFT: Robust Supervised Fine-tuning for Large Language Models under Noisy Response

**ArXiv:** [2412.14922](https://arxiv.org/abs/2412.14922)

## 🎯 Pitch

ROBUSTFT introduces a novel supervised fine-tuning framework that robustly adapts large language models (LLMs) using real-world datasets contaminated with noisy responses, by combining multi-expert collaborative noise detection, context-enhanced relabeling, and entropy-based data selection. This innovation enables LLMs to maintain high performance even as noise levels soar—significantly outperforming standard methods—making domain adaptation with imperfect data practical and reliable for critical applications like healthcare, finance, and beyond.

---

## 1. Executive Summary
ROBUSTFT is a supervised fine‑tuning framework that automatically detects and repairs noisy training examples for large language models (LLMs) by combining multi‑expert agreement, context‑based relabeling, and entropy‑based data selection. Across five datasets and three model families, it prevents the steep performance collapse caused by noisy supervision and yields consistent gains over both vanilla models and standard SFT, especially under heavy noise (e.g., on MMLU at 70% noise, accuracy rises from 37.3 with SFT to 67.6; Table 1).

## 2. Context and Motivation
- Problem addressed:
  - Supervised fine‑tuning (SFT) relies on high‑quality instruction–response pairs. In real datasets, labels/responses often contain noise from annotation mistakes and model hallucinations, degrading downstream performance.
  - Figure 1 quantifies the risk: with 30% noisy training data, the MMLU score drops by 8.9% relative to the vanilla LLM baseline; further noise causes sharper declines.

- Why this matters:
  - Practically, many organizations adapt general LLMs to domain tasks (medical, finance, reasoning) using in‑house data that is imperfect. Robustness to noise makes such adaptation viable without expensive re‑annotation.
  - Methodologically, most “noisy label learning” research targets classification with a small set of discrete labels. LLM SFT involves open‑ended text generation, where noise is harder to spot and correct.

- Shortcomings of prior approaches:
  - Classical noisy‑label methods (Section 5.1) assume discrete labels and do not exploit rich context in textual responses.
  - LLM self‑filtering is unreliable: Section 4.2.1 shows the SelfSelect baseline—which asks the model to pick good data—underperforms even plain SFT on several datasets.
  - Enhanced SFT with external instruction sets (e.g., Hermes‑3, Tulu‑3; Table 1) does not reliably improve downstream adaptation under noise.
  - Retrieval‑augmented inference methods (SelfRAG) help modestly at inference time but do not fix corrupted training signals (Table 1).

- Positioning:
  - ROBUSTFT (Sections 3.1–3.5) is a self‑contained, two‑stage framework tailored to open‑ended LLM SFT: (1) multi‑view noise detection via agreement among expert predictions; (2) denoising by context‑enhanced relabeling plus entropy‑based sample selection, followed by SFT on the curated set.

## 3. Technical Approach
The framework (Figure 2, Algorithm 1) operates in three phases. Key notation is introduced in Section 3.

1) Noise detection by multi‑expert agreement (Section 3.2)
- Step A: Base prediction. For each training query `q_i`, the base LLM `M` predicts a response `ŷ_i = M(q_i)` (Equation 1).
- Step B: Reasoning‑enhanced prediction. A “reasoning‑enhanced LLM” iterates between explicit reasoning (`M_Reas`) and self‑reflection (`M_Refl`) to produce a final answer `ŷ_i^reas` (Equation 2). Conceptually, this is a loop: explain the answer step‑by‑step, critique the reasoning, then refine.
- Step C: Consistency checking. A `Checker` compares three sources: the dataset label `y_i`, the base prediction `ŷ_i`, and the reasoning‑enhanced prediction `ŷ_i^reas`. It emits `r_i ∈ {0,1}` (Equation 3): `r_i=1` means high agreement (clean), `r_i=0` means disagreement (potential noise). The paper sets `n=4` and an agreement threshold `θ=50%` during implementation (Section 4.1.3), implying a simple majority rule across multiple expert outputs/prompts.
- Output: Split the data into `D_clean` (reliable) and `D_noise` (potentially noisy).

Why this works: relying on a single LLM prediction is brittle because LLMs can hallucinate confidently. Agreement between a plain prediction, a reasoning‑plus‑reflection prediction, and the original label offers a stronger signal that the training pair is trustworthy.

2) Denoising via context‑enhanced relabeling (Section 3.3)
- Step A: Build a retrieval index over clean examples. Each query `q` is mapped to an embedding `h = Encoder(q) ∈ R^d` (Equation 4).
- Step B: For each noisy sample `(q_i, y_i) ∈ D_noise`, retrieve the `k` most similar clean query–answer pairs from `D_clean` and condition the LLM on these as context. Generate a “context‑enhanced” answer `ŷ_i^cont` (Equation 5). Intuition: showing the model similar vetted examples helps it produce a better label for the noisy case.
- Step C: Review and synthesize. A `Review` agent compares the context‑enhanced answer `ŷ_i^cont` with the reasoning‑enhanced answer `ŷ_i^reas` and consolidates them into a final relabeled answer `ỹ_i` (Equation 6). This adds another layer of cross‑checking before accepting a self‑generated label.
- Output: a denoised set `D_denoise = {(q_i, ỹ_i)}`.

3) Confidence‑based selection by entropy (Section 3.4)
- Step A: Compute sequence‑level uncertainty for each context‑enhanced generation using normalized token entropy `H(ŷ_i^cont)` (Equation 7). Low entropy indicates the model is confident (more deterministic token probabilities).
- Step B: Keep only the most confident fraction. Rank by `H(ŷ_i^cont)` and retain the top‑`β` fraction (lowest entropy) to form `D_select` (Equation 8). The default is `β = 50%` (Section 3.4; sensitivity in Figure 3).

4) Final SFT on curated data (Section 3.5)
- Concatenate the high‑trust original data and the vetted relabels: `D_ft = D_clean ∪ D_select`.
- Fine‑tune the LLM by standard language‑modeling loss on `D_ft` (Equation 9). Implementation uses LoRA (low‑rank adapters) for parameter‑efficient SFT over 2 epochs (Section 4.1.3).

Design choices and rationale
- Multi‑view detection: pairs that survive agreement between `y_i`, `ŷ_i`, and `ŷ_i^reas` are likely clean. Disagreements trigger careful relabeling rather than outright deletion, preserving data volume.
- Context‑enhanced relabeling: retrieving similar clean examples provides grounded, in‑distribution references, reducing hallucination risk during relabeling.
- Entropy filtering: even after relabeling, some self‑annotations are uncertain. Filtering by entropy mitigates error propagation in the final SFT.

## 4. Key Insights and Innovations
- Multi‑expert, reasoning‑aware noise detection (fundamental)
  - Novelty: combines a base LLM, a reasoning‑and‑reflection LLM, and a consistency `Checker` (Equations 1–3) to identify suspicious samples. This exploits complementary strengths—direct answer vs. reasoned answer—and avoids overreliance on any single predictor.
  - Evidence: Removing the `Checker` degrades MMLU accuracy from 68.2 to 65.3 at 30% noise (Table 3), showing that agreement‑based detection is a major driver.

- Context‑enhanced relabeling + review (fundamental)
  - Novelty: instead of discarding noisy items, the method repairs them by conditioning on nearest clean exemplars (Equation 5) and then consolidating with a `Review` agent (Equation 6). This leverages the dataset’s own verified knowledge.
  - Evidence: Removing context‑enhanced relabeling (`w/o CER`) drops MMLU from 68.2 to 67.7 and ARC from 84.9 to 84.1 (Table 3). Removing the reasoning‑enhanced expert (`w/o REL`) yields similar declines (67.4 and 84.1), indicating both inputs to the reviewer matter.

- Entropy‑based selection to prevent bad self‑labels (important incremental)
  - Novelty: ranks relabeled items by generation entropy (Equation 7) and keeps only the most confident half (Equation 8). This adds a probabilistic quality gate specific to open‑ended generation.
  - Evidence: `w/o Selection` produces the largest single ablation drop (MMLU −2.5, ARC −1.7 at 30% noise; Table 3). Sensitivity in Figure 3 shows performance peaks around `β=40–50%`.

- Self‑contained pipeline for open‑ended SFT (important incremental)
  - The system avoids dependence on external gold annotations during denoising and acts directly on free‑form responses. Section 3.5 and Algorithm 1 present an end‑to‑end process designed for LLM SFT, contrasting with classification‑oriented noise methods referenced in Section 5.1.

## 5. Experimental Analysis
- Evaluation setup (Section 4.1)
  - Datasets spanning general knowledge and domain tasks: MMLU, ARC, PubMedQA (biomedical), DROP (numeric/reading), FPB (financial).
  - Noise regime: inject 30%, 50%, 70% noisy answers into training sets (details of injection are in the appendix per Section 4.1.3 summary).
  - Models: Llama‑3.1‑8B, Llama‑3.2‑3B, Gemma‑2‑9B (Table 2).
  - Training: LoRA fine‑tuning for 2 epochs using Llama‑Factory; default `β=50%`, context length `k=3`, `n=4` experts, agreement threshold `θ=50%` (Section 4.1.3; Figure 3 studies `β` and `k`).

- Baselines (Section 4.1.2)
  - `Vanilla`: zero SFT inference.
  - `SFT`: plain fine‑tuning on the noisy data.
  - SFT with external corpora: `Hermes‑3`, `Tulu‑3`.
  - Denoising/selection: `NoiseAL`, `SelfLabel`, `SelfSelect`.
  - Inference augmentation: `SelfRAG`.

- Headline results (Llama‑3.1‑8B; Table 1)
  - MMLU: ROBUSTFT 68.2/68.0/67.6 (noise 30/50/70), beating `Vanilla` 65.3 and far exceeding `SFT` 59.5/47.5/37.3. Reported relative gains over SFT: +14.6%, +43.2%, +81.2%.
  - ARC: 84.9/84.7/84.1 vs 82.7 (`Vanilla`) and 70.7/61.7/47.5 (`SFT`).
  - PubMedQA: 75.8/75.6/75.0 vs 72.0 (Vanilla) and 66.4/36.7/32.8 (SFT).
  - DROP: 90.3/88.5/87.9 vs 87.2 (Vanilla) and 85.3/78.6/66.4 (SFT).
  - FPB: 84.4/80.5/76.2 vs 75.5 (Vanilla) and 79.7/58.4/34.9 (SFT).
  - Quote:
    > Table 1: “Increasing noise levels deteriorate SFT sharply, while ROBUSTFT improves over Vanilla by up to +11.8 points (FPB 30%) and over SFT by up to +129% relative (PubMedQA 70%).”

- Cross‑model generality (Table 2)
  - Llama‑3.2‑3B: MMLU 58.5/58.2/57.9 vs Vanilla 54.9 and SFT 55.0/48.4/38.3.
  - Gemma‑2‑9B: MMLU 72.5/72.1/71.3 vs Vanilla 70.3 and SFT 63.6/52.1/40.3; FPB at 70% noise jumps from SFT 35.6 to ROBUSTFT 87.7.
  - Note: Table 2’s FPB 70% for Llama‑3.1‑8B is 73.2, slightly below Table 1’s 76.2, indicating minor run/config variability; the trend remains consistent.

- Robustness and diagnostics
  - Ablations (Table 3): every component contributes; `Selection` and `Checker` drive the largest gains. Removing `Reviewer` causes smaller but consistent drops.
  - Sensitivity (Figure 3): best `β` around 40–50%; performance plateaus for context length `k=3–5`, suggesting a few similar examples suffice for relabeling.
  - Perplexity analysis (Figure 4): with noise, both vanilla and SFT show higher, more dispersed perplexity; ROBUSTFT concentrates density at lower perplexity across noise levels and datasets, indicating more confident predictions.
  - Category‑wise (Figure 5): noise harms knowledge‑heavy domains (History, Health, Law) the most; ROBUSTFT lifts scores broadly rather than in isolated categories.
  - Stability (Figure 6): under instruction paraphrasing (five runs using GPT‑4o to rephrase prompts), accuracy remains steady with small variance, including at high noise.

- Do the experiments support the claims?
  - Yes: Results repeatedly show that standard SFT collapses under noise, whereas ROBUSTFT not only avoids collapse but often outperforms the un‑fine‑tuned base model. Gains hold across datasets, model sizes, and noise levels, and ablations attribute gains to the proposed components.

## 6. Limitations and Trade-offs
- Dependence on partial cleanliness:
  - The method assumes the `Checker` can carve out a meaningful `D_clean` from the noisy corpus. If nearly all samples disagree (e.g., systematic label corruption), retrieval and relabeling may lack reliable anchors.

- Checker details and thresholds:
  - The `Checker` is defined as a consistency function (Equation 3) with `n=4` and `θ=50%` in implementation (Section 4.1.3), but its exact agreement metric and expert composition are not extensively specified. Different implementations may change which samples are marked clean/noisy.

- Computational overhead:
  - Compared with plain SFT, ROBUSTFT runs multiple inference passes per sample: base generation, reasoning+reflection loop, retrieval‑conditioned generation, and review; plus embedding/indexing and entropy computation. This raises preprocessing cost and may be substantial for very large corpora.

- Retrieval and encoder choice:
  - Relabeling quality depends on retrieving semantically relevant clean examples (Equation 5). The paper abstracts `Encoder(·)` (Equation 4); performance could vary with embedding choice and domain shift.

- Entropy metric scope:
  - Entropy is computed on the context‑enhanced output (Equation 7). Confident but wrong generations can slip through; conversely, correct but diverse phrasing may be filtered out. This is partly mitigated by keeping only the top‑`β` fraction, but it remains a heuristic.

- Noise model and data transparency:
  - Section 4.1 mentions “introducing varying degrees of noise perturbation,” but the main text does not detail the corruption process. Different noise types (adversarial vs. random vs. systematic bias) could affect outcomes.

- Scope of post‑training objectives:
  - The framework targets SFT with next‑token loss (Equation 9). It does not integrate with preference learning or RLHF, where noise manifests differently (e.g., in preference pairs).

## 7. Implications and Future Directions
- Impact on practice:
  - The paper provides an actionable recipe for organizations to fine‑tune LLMs on imperfect real‑world corpora. A drop‑in pre‑SFT curation step—agreement‑based detection, retrieval‑guided relabeling, entropy filtering—can stabilize and often improve over the vanilla model under heavy noise.

- Research directions:
  - Stronger agreement models: replace the binary `Checker` with calibrated uncertainty estimation, semantic‑equivalence scoring, or multiple heterogeneous LLMs with learned weights.
  - Better relabeling governance: incorporate verifier models or lightweight human spot‑checks for high‑impact samples; add contradiction tests between `ŷ_i^cont` and `ŷ_i^reas`.
  - Beyond entropy: combine entropy with semantic confidence (e.g., entailment scores, self‑consistency voting, or edit distance to retrieved exemplars).
  - Noise typology studies: evaluate under adversarial and systematically biased noises; analyze cross‑lingual and code‑generation settings.
  - Integration with other post‑training methods: apply ROBUSTFT as a data curation front‑end for preference optimization and RLHF, where misplaced preferences are common.

- Applications:
  - Domain adaptation with weak supervision (medical Q&A, finance analysis, legal reasoning).
  - Bootstrapping specialized assistants from mixed‑quality logs or scraped data.
  - Continual learning where on‑the‑fly relabeling and selection counter drift and accumulation of errors.

Overall, ROBUSTFT reframes noisy SFT as a detect‑then‑repair problem using multi‑expert agreement and context‑aware relabeling, then enforces quality with entropy‑based selection. The combination is simple, self‑contained, and empirically effective across diverse tasks and models (Tables 1–2; Figures 3–6), making it a practical baseline for noise‑robust LLM adaptation.
