# Gemma 3 Technical Report

**ArXiv:** [2503.19786](https://arxiv.org/abs/2503.19786)
**Authors:** Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean‑bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert Busa‑Fekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, Jan‑Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, András György, André Susano Pinto, Anil Das, Ankur Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. Choquette‑Choo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak‑Plucińska, Harman Singh, Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean Pouget‑Abadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju‑yeong Ji, Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine, Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth, Siim Põder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Jessica Lo, Erica Moreira, Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Jean‑Baptiste Alayrac, Rohan Anil, Dmitry Lepikhin, Sebastian Borgeaud, Olivier Bachem, Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, Léonard Hussenot
**Institutions:** Google DeepMind

## 🎯 Pitch

Gemma 3 introduces a family of efficient multimodal models that masterfully combine vision and language understanding while drastically optimizing memory use, thanks to an innovative interleaved attention architecture. This advancement enables real-world deployment on edge devices, expanding accessibility for long-document and multilingual processing, and enhances practical applications such as large-document analysis and reliable coding assistance.

---

## 1. Executive Summary (2-3 sentences)
Gemma 3 is a family of small-to-mid-size open multimodal language models (1B–27B parameters) that add strong image understanding, much longer context (up to 128K tokens), and broader multilingual coverage while keeping consumer-grade deployment in mind. The core technical advance is an architecture and training recipe that sharply reduces inference memory for long contexts by interleaving many local-attention layers with sparse global-attention layers, coupled with distillation- and RL-based post-training that lifts math, coding, and chat performance to the level of much larger prior models (e.g., Table 6; Table 5).

## 2. Context and Motivation
- Problem/gap:
  - Long-context models suffer from exploding inference memory due to the `KV cache` (the stored “keys” and “values” that attention reuses for prior tokens). This becomes prohibitive beyond tens of thousands of tokens, especially on consumer GPUs.
  - Many “small” open models lack competitive instruction following, math/coding reasoning, and true multimodality (vision+text) while remaining efficient to deploy.
  - Non-English performance remains uneven in many open models.
- Why it matters:
  - Long documents, codebases, and multi-file contexts are common real-world workloads. Efficient long-context models unlock applications like large-document analysis, EHR review, and long transcript reasoning on edge devices.
  - Reliable math/coding is essential for practical assistant tasks; visual reasoning (reading, charts, documents) is increasingly a baseline capability.
- Where prior approaches fall short:
  - “Global-only” attention (full attention in every layer) drives KV cache growth linearly with context length and number of layers; memory dominates weights at long context (Figures 5–6).
  - Visual encoders at fixed square resolutions (e.g., 896×896) can badly distort non-square images, hurting OCR and small-object detection.
  - Post-training recipes for small models often trail large closed models on math/coding/reasoning and multilingual breadth.
- Positioning:
  - Gemma 3 changes the backbone to interleave many `local sliding-window` layers with fewer `global` layers (Section 2; “5:1 interleaving of local/global layers”), raises global RoPE base frequency for 128K context, and introduces an inference-time `Pan & Scan` (P&S) scheme to preserve image aspect ratio (Section 2.1).
  - It pairs that with distillation-based pre-training and a new instruction-tuning approach using reward-model averaging and RL (Section 3), yielding small models that rival much larger predecessors (Table 6) and score among the top open models in human preference (Table 5).

## 3. Technical Approach
Step-by-step overview of what Gemma 3 is and how it works:

- Model family and sizes (Table 1):
  - Four dense models: `1B`, `4B`, `12B`, `27B`.
  - Vision capability via a shared `SigLIP` encoder (400M parameters) for 4B/12B/27B; the `1B` model is text-only (Vision Encoder: 0 in Table 1).
  - Tokenizer: 262k-vocab SentencePiece with split digits, preserved whitespace, and byte-level encodings (Section 2.2 “Tokenizer”).

- Long-context architecture (Section 2; Figures 3–6):
  - Interleaved attention: a fixed pattern of 5 local sliding-window self-attention layers for every 1 global attention layer; the model starts with a local layer (Section 2 “5:1 interleaving…”).
    - `Local attention` here means each token attends only to a recently preceding span (window). Gemma 3 uses a short local span of 1024 tokens.
    - `Global attention` layers retain full attention over the sequence and are responsible for propagating long-range dependencies.
  - RoPE configuration for long context (Section “Long context”):
    - `RoPE` (rotary position embeddings) base frequency is increased from 10k to 1M for global layers to preserve attention geometry over very long ranges, while local layers keep 10k.
    - Training is done at 32K contexts, then the 4B/12B/27B models are rescaled (“positional interpolation”) to 128K at the end of pre-training; an 8× scaling factor is used (Figure 7).
  - Why it helps:
    - KV cache memory scales with the number of layers that maintain full-sequence keys/values. By making most layers local with a short window, KV memory is slashed while keeping periodic global layers as “highways” to carry long-range information.

- Vision input pathway (Section 2.1; Tables 7–8; Figure 1):
  - A frozen `SigLIP` encoder (400M ViT trained with CLIP-like loss) converts each image (resized to 896×896) into a fixed set of 256 “soft tokens” (vision embeddings).
  - To reduce artifacts from the fixed square resolution, `Pan & Scan` (P&S) is applied at inference time only: the image is segmented into non-overlapping crops to preserve aspect ratios and small details; each crop is resized to 896×896 and encoded, with a cap on the number of crops (Section “Pan & Scan”).
  - Encoder resolution matters: higher resolution improves OCR and document VQA even though outputs are pooled to 256 tokens (Table 7 shows DocVQA improving 31.9→59.8 when encoder input goes 256→896).

- Pre-training (Section 2.2):
  - Data scale and mixture:
    - Token budgets: 27B model on 14T tokens; 12B on 12T; 4B on 4T; 1B on 2T (Section “Training data”).
    - Mixed text+image data; expanded multilingual coverage using strategies inspired by UniMax sampling (Chung et al., 2023).
  - Distillation procedure:
    - For each token, 256 candidate logits are sampled with probabilities weighted by a large “teacher” model; a cross-entropy loss is computed after renormalizing the sampled distribution while setting all non-sampled logits to zero probability (Section “Distillation”).
    - Ablation (Figure 8) shows a nuanced teacher-size effect: smaller teachers help on very short training horizons, but with longer training, a larger teacher yields better perplexity.

- Post-training (Instruction Tuning; Section 3; Table 4):
  - SFT via improved knowledge distillation from a large instruction-tuned teacher, then RL fine-tuning using:
    - `BOND` (best-of-N distillation), `WARM` (weight-averaged reward models), and `WARP` (weight-averaged rewarded policies).
    - Rewards blend human preference models, code execution feedback, and ground-truth solutions for math datasets (Section “Reinforcement learning objectives”).
  - Data filtering removes unsafe or personal information, reduces hallucinations, and incorporates examples encouraging refusals and better attribution (Section “Data filtering”).
  - Chat formatting uses explicit control tokens and requires a `[BOS]` token at the beginning; IT models terminate with `<end_of_turn>` (Table 4).

- Quantization-aware training (QAT; Section 2.3; Table 3):
  - Each model is fine-tuned ~5,000 steps with QAT to produce weight-quantized checkpoints (per-channel int4, per-block int4, and switched-FP8).
  - Memory impact at 32K context (weights+KV):
    - Example 27B: bf16 72.7 GB vs int4 32.8 GB vs SFP8 46.1 GB (Table 3).
  - This lowers the barrier for on-device or single-GPU inference.

- Training system (Section 2.4; Table 2):
  - Trained on TPUv4, TPUv5e, TPUv5p using JAX/Pathways with ZeRO-3-style optimizer sharding; multi-pod data replica reductions; vision embeddings are precomputed so vision adds no runtime cost during LLM training.

Analogy for the attention design:
- Think of local layers as “neighborhood streets” handling short-range interactions cheaply, while every sixth layer is a “highway” that lets information jump anywhere in the document. Fewer highways reduce the memory needed to keep the entire map in view, but still allow rapid long-distance travel when needed.

## 4. Key Insights and Innovations
- Memory-efficient long context via interleaved local/global attention:
  - What’s new: a fixed 5:1 local:global ratio with short local windows (1024) ensures only a subset of layers store long-range KV states, reducing KV cache growth (Section 2; Figures 5–6).
  - Why it matters: Figure 5 shows that at 32K context, “global only” has ~60% memory overhead from KV cache, but with `1:3` and window 1024, overhead drops below ~15%. Figure 6 shows KV memory rising far more slowly with context under the Gemma 3 scheme than “global only.”
  - Difference vs prior: Gemma 2 and LLaMA-like models used full (global) attention in more layers (Gemma 2 used 1:1 with larger windows); Gemma 3 demonstrates perplexity barely changes even when making local layers dominant (Figures 3–4).

- Practical, high-quality vision with fixed 256-token image embeddings and Pan & Scan:
  - What’s new: A frozen SigLIP encoder emitting a fixed 256-vector representation per image, augmented at inference by P&S windowing to preserve aspect ratio when needed (Section 2.1).
  - Why it matters: It keeps visual compute predictable and small, yet preserves OCR/small-detail performance when P&S is enabled. Table 8 shows large boosts from enabling P&S on tasks requiring text reading (e.g., 27B IT: InfoVQA +17.0 points).

- Distillation-first training with RL post-tuning that uplifts small models:
  - What’s new: A post-training stack blending on-policy distillation and policy/reward model averaging (BOND, WARM, WARP), with code execution and math ground-truth rewards (Section 3).
  - Why it matters: The 4B IT model reaches or surpasses Gemma 2 27B IT on several benchmarks (Table 6), showing that improved post-training can compress capabilities into smaller models.

- Safe, responsible release with markedly lower memorization rates:
  - What’s new: A systematic memorization audit shows Gemma 3 memorizes long-form text far less than prior Gemma/Gemini models (Figure 9), with most memorization being approximate rather than verbatim; none of the memorized outputs contain personal information per Google Cloud SDP scanning (Section 6).
  - Why it matters: Lower memorization reduces privacy risk and unintended copyright leakage in open-weight releases.

These are fundamental innovations (long-context memory design; P&S integration) combined with strong, well-engineered improvements (distillation+RL recipe; QAT and deployment format coverage).

## 5. Experimental Analysis
- Evaluation methodology and setup:
  - Human preference: LMSYS Chatbot Arena blind A/B tests yield Elo ratings across many models (Table 5).
  - Static benchmarks: A broad suite spans general knowledge, multilingual QA/translation, math/coding, and multimodal VQA (Tables 6, 9–18; Figures 2, 7–8). Long-context is measured with `RULER` and `MRCR` at 32K and 128K (Table 15). Multimodal includes OCR-heavy tasks (DocVQA, InfoVQA, TextVQA), diagram/chart QA (AI2D, ChartQA), and MMMU (Tables 11–12, 16–17).
  - Metrics: Accuracy, pass@1, CHRF/ANLS/CIDEr depending on task; consistent few-shot or zero-shot settings are summarized in Tables 19–21.

- Headline results:
  - Chatbot Arena (human preference):
    - > “Gemma-3-27B-IT 1338 Elo… among the top 10” and above many non-thinking open models (Table 5). This is notable for a 27B dense model competing with larger or closed models.
  - General and STEM (instruction-tuned; Table 6):
    - `27B IT` reaches MMLU-Pro 67.5, MATH 89.0, Global MMLU-Lite 75.1, LiveCodeBench 29.7. The `4B IT` attains MATH 83.8 and LiveCodeBench 24.6, rivalling or surpassing Gemma 2 27B IT.
  - Code and math (instruction-tuned; Table 18):
    - `27B IT` scores HumanEval 87.8 pass@1, GSM8K 95.9, MATH 89.0, LiveCodeBench 39.0; `12B IT` is close (HumanEval 85.4, GSM8K 94.4). These gains align with the RL/data strategies (Section 3).
  - Multimodal (pre-trained and IT; Tables 11–12, 16):
    - Pre-trained `27B` achieves DocVQA 85.6, InfoVQA 59.4, TextVQA 68.6 (Table 11); after fine-tuning, IT `27B` reaches InfoVQA 70.6, ChartQA 78.0, and MathVista (testmini) 67.6, with P&S active (Table 16).
    - P&S boosts OCR-like tasks significantly (Table 8), confirming the inference-time windowing is effective.
  - Long-context (Table 15; Figure 7):
    - Pre-trained `27B` gets RULER 32K 85.9 and 128K 72.9; IT `27B` keeps RULER 32K 91.1 but drops at 128K to 66.0. Figure 7 shows perplexity generalizes well to 128K after RoPE rescaling but degrades beyond that.

- Ablations and robustness:
  - Local:Global ratio (Figure 3): Perplexity changes minimally even at 7:1 local:global; strong evidence that many local layers do not harm modeling quality.
  - Sliding window size (Figure 4): Windows can be reduced considerably (e.g., 1024) without meaningful perplexity penalty.
  - KV memory (Figures 5–6): Clear reduction versus “global only,” with much better scaling at longer contexts.
  - Teacher size vs training horizon (Figure 8): Large teachers benefit longer training; small teachers can help short runs.
  - Vision resolution (Table 7) and P&S (Table 8): Higher encoder resolution and P&S produce sizable gains on OCR/document QA.

- Do experiments support claims?
  - Memory/efficiency: Figures 5–6 provide direct measurements of KV vs weight memory across configurations; the interleaving strategy delivers the promised KV reduction.
  - Capability: Table 6 shows Gemma 3 IT models outperform Gemma 2 IT across domains; Table 5 confirms improved human preference.
  - Vision: Tables 7–8 and 16 demonstrate the value of higher encoder resolution and P&S for OCR-like tasks; overall multimodal performance is competitive (not always state-of-the-art on COCO/VQAv2 vs PaliGemma 2; Table 12).
  - Safety/memorization: Figure 9 shows an order-of-magnitude reduction in memorization rates vs prior models; SDP scan detected no personal data in memorized outputs.

- Mixed/conditional findings and trade-offs:
  - IT long-context performance at 128K is lower than pre-trained (Table 15), suggesting some degradation from instruction tuning for extreme lengths.
  - P&S improves OCR-like tasks but adds inference overhead and must be capped for speed (Section 2.1 “Pan & Scan”).
  - On certain vision benchmarks (e.g., VQAv2, COCO caption), PaliGemma 2 remains slightly ahead (Table 12), likely reflecting differences in pretraining corpora/objectives.

## 6. Limitations and Trade-offs
- Architectural assumptions:
  - The 5:1 local:global pattern assumes periodic global layers suffice to propagate long-range dependencies. While perplexity and many tasks hold up, worst-case tasks that require dense long-range attention at every layer might suffer (not directly tested).
- Context scaling:
  - Models are trained at 32K and rescaled to 128K via RoPE interpolation. Figure 7 shows good generalization to 128K but rapid degradation beyond; this approach may not extend reliably past 128K without retraining.
  - `1B` model supports only 32K context (Section “Long context”).
- Instruction-tuning vs length:
  - IT models show weaker scores at 128K vs pre-trained (Table 15), hinting at a trade-off between conversational alignment and extreme long-context robustness.
- Vision pathway constraints:
  - The encoder is frozen and operates at 896×896 with average pooling to 256 tokens. While efficient, this caps granularity; very fine-grained or high-res tasks can require many P&S crops, increasing latency (Section 2.1).
- Compute and data:
  - Training uses large TPU clusters (Table 2) and trillions of tokens (Section 2.2). Reproducing the full recipe may be out of reach for small labs even if inference is lightweight.
- Evaluation caveats:
  - Table 6 avoids direct comparisons to external models due to evaluation setting mismatches; claims rely on internal consistency and third-party leaderboards for context.
  - Chatbot Arena Elo (Table 5) is preference-based and does not capture multimodal abilities, which are a key selling point of Gemma 3.
- Safety:
  - The SDP-based personal information detection is high-recall but coarse; “no PI in memorized outputs” is bounded by detection thresholds and may include false negatives (Section 6).
  - Baseline safety evaluations report low violation rates but details and datasets are not fully disclosed (Section 7.3).

## 7. Implications and Future Directions
- Landscape impact:
  - Demonstrates that careful architectural choices (interleaved local/global attention) can make 128K context feasible with small memory overhead on mid-size dense models (Figures 5–6), moving long-context capability from boutique models to broadly deployable open checkpoints.
  - Validates that post-training (distillation + RL with code execution and math ground truth) can compress capabilities that once required 70B–400B models into 4B–27B models (Table 6; Table 18).
  - Offers a pragmatic multimodal recipe (frozen SigLIP→256 tokens + P&S) that balances efficiency and OCR/document performance (Tables 7–8, 16).

- Follow-up research enabled/suggested:
  - Better alignment for ultra-long inputs: mitigate the 128K IT drop (Table 15) via length-aware RLHF, long-sequence SFT, or curriculum sampling during post-training.
  - Adaptive attention schedules: learn where global layers are needed rather than fixed 5:1 interleaving; dynamic window sizes conditioned on input structure.
  - Train-time multimodal fusion: explore unfreezing or co-training the vision encoder, or cross-attention adapters, to close the small gaps on COCO/VQAv2 noted in Table 12.
  - Efficient P&S variants: learned crop policies or content-aware tiling to reduce the number of crops while preserving small text and details.
  - Beyond 128K: explicit long-sequence pretraining blocks, hybrid recurrent memory, or sparse retrieval-augmented caches for million-token contexts.
  - Safety and privacy: stronger PI detection beyond SDP, red-teaming multimodal jailbreaks, and formal memorization bounds under distillation.

- Practical applications:
  - On-device and edge assistants for document understanding (invoices, forms, medical records), given fixed 256-token image embeddings and QAT checkpoints (Table 3).
  - Developer tooling for long codebases, leveraging 128K context and high HumanEval/LiveCodeBench scores (Tables 6, 18).
  - Multilingual chat and reasoning assistants with improved non-English coverage (Table 6; Appendix Tables 13–14).
  - Vision assistants for OCR-heavy tasks (DocVQA, InfoVQA), with P&S enabled (Table 8; Table 16).

> Key evidence to remember:
> - KV cache overhead plummets with few global layers and small windows (Figures 5–6).
> - 4B IT performance rivals or exceeds prior 27B IT on many tasks (Table 6), showing the potency of the new post-training recipe.
> - Multimodal performance is strong and benefits substantially from P&S on text-in-image tasks (Table 8; Table 16).
> - Memorization rates are markedly lower than prior generations, with no detected PI in memorized outputs (Figure 9).

Overall, Gemma 3’s design choices make long-context multimodal LLMs more practical in small footprints without sacrificing capability, and they provide a clear blueprint—architectural, training, and deployment—that others can adapt or extend.
