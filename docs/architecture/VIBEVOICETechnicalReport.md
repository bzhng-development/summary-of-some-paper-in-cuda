# VIBEVOICE Technical Report

**ArXiv:** [2508.19205](https://arxiv.org/abs/2508.19205)

## 🎯 Pitch

VIBEVOICE introduces a groundbreaking framework for long-form, multi-speaker text-to-speech synthesis by combining a large language model with a novel, ultra-compressed, continuous speech tokenizer and an autoregressive next-token diffusion decoder. This enables the system to generate up to 90 minutes of high-fidelity, natural-sounding conversational audio with up to four distinct voices, all within a single context. By preserving realistic conversational flow and dramatically improving computational efficiency, VIBEVOICE sets a new state-of-the-art for scalable, nuanced, and engaging audio generation—surpassing both open and proprietary baselines in human evaluations of preference, realism, and richness.

---

## 1. Executive Summary
VIBEVOICE is a long-form, multi‑speaker text‑to‑speech (TTS) system that combines a large language model with a token‑level diffusion decoder and a new ultra‑compressed, causal speech tokenizer. It synthesizes up to 90 minutes of conversational audio with up to four speakers in a single run and, on subjective human evaluations, outperforms strong open and closed baselines on preference, realism, and richness (Figure 1; Table 1).

## 2. Context and Motivation
- Problem addressed
  - Generating natural, long conversational audio (e.g., podcasts, multi‑narrator audiobooks) is difficult. Stitching together short utterances misses turn‑taking, conversational flow, and context‑aware prosody (Section 1).
  - Existing systems either are not open‑sourced or struggle with stability and length for multi‑speaker conversations [Goo24, PSJ+24, Nar25, Ses25, LWI+24, ZQW+25] (Section 1).

- Why it matters
  - Practical: high‑quality podcast‑like generation, scripted dialogues, and long narrative content creation.
  - Scientific: tests whether large‑context sequence modeling plus continuous (rather than discrete) acoustic targets can scale to hour‑long synthesis without collapsing quality.

- Prior approaches and gaps
  - Single‑speaker, short‑utterance TTS has progressed rapidly (e.g., NaturalSpeech‑2, Voicebox, Seed‑TTS; Section 1), but these models do not target multi‑speaker, hour‑scale conversation.
  - Multi‑speaker long‑form work has appeared recently, but faces length/stability limits or is not open (Section 1).

- Positioning of this work
  - Introduces a unified, autoregressive “next‑token diffusion” framework (Figure 2) paired with a new continuous, causal acoustic tokenizer operating at 7.5 Hz (3200× compression from 24 kHz; Section 2.1). This drastically shortens the sequence the model must generate, enabling 90‑minute contexts while preserving quality (Introduction; Section 2).

## 3. Technical Approach
At a high level, VIBEVOICE turns a multi‑speaker script and a set of short voice prompts into a single long waveform by:
1) encoding prompts and, during generation, already‑synthesized audio with two tokenizers,
2) running an LLM over an interleaved sequence of role tags, text, and speech features, and
3) using a diffusion head to predict continuous acoustic latents token‑by‑token, which a decoder converts to audio (Figure 2; Section 2.2).

Key components and how they work:

- Two complementary speech tokenizers (Section 2.1)
  - Acoustic tokenizer (σ‑VAE)
    - Goal: compress audio into continuous latent vectors that preserve timbre, prosody, and fidelity.
    - σ‑VAE idea: A VAE variant where the latent variance is not learned per input; instead, the model samples σ from a fixed distribution N(0, Cσ). This helps avoid “variance collapse” when such latents are later predicted autoregressively (Section 2.1).
    - Latent sampling: z = μ + σ ⊙ ε, with ε ~ N(0, 1) and σ ~ N(0, Cσ).
    - Architecture: a mirror‑symmetric encoder–decoder with 7 hierarchical stages. Modified Transformer blocks use 1D depth‑wise causal convolutions instead of self‑attention (for streaming). Six downsampling layers give an overall 3200× reduction from 24 kHz to 7.5 tokens per second (Section 2.1).
    - Scale: each encoder/decoder is ~340M parameters. Training uses DAC‑style adversarial/discriminator losses for reconstruction quality [KSL+23] (Section 2.1).
  - Semantic tokenizer
    - Goal: encode content (what is being said) deterministically.
    - Architecture mirrors the acoustic encoder, but without the VAE. It is trained via an ASR proxy task: the encoder’s output is decoded by a Transformer to predict transcripts; the decoder is discarded after pre‑training (Section 2.1).

- Input representation and sequence modeling (Section 2.2)
  - The model concatenates, as a single sequence, per‑speaker voice prompts and text scripts, each prefixed with role identifiers:
    - X = [Speaker1: z1, …, SpeakerN: zN] + [Speaker1: T1, …, SpeakerN: TN]
    - `zk` are acoustic latent features (voice “font” from prompts). `Tk` are per‑speaker text.
  - During generation, produced speech segments are encoded on the fly by both tokenizers to form a hybrid (acoustic + semantic) context for future steps (Section 2.2).
  - LLM backbone: Qwen2.5 at 1.5B and 7B parameters (Section 2.2).
  - Training strategy: tokenizers are frozen; only the LLM and a small diffusion head are trained. Input sequence length is increased via curriculum from 4,096 to 65,536 tokens to teach long‑context handling (Section 2.2).
  - Important ratio: the 7.5 Hz acoustic tokens produce a speech‑to‑text token ratio of about 2:1 (two speech tokens per BPE text token), which keeps long sequences computationally tractable (Section 1).

- Token‑level next‑token diffusion for acoustic prediction (Section 2.2)
  - What is “diffusion”? A generative process that learns to reverse a gradual noising process; at inference, iterative denoising turns noise into a sample.
  - Here, the diffusion head (4 layers) predicts the acoustic VAE vector `za,i` for the next token i, conditioned on the LLM hidden state `hi` at that position (Figure 2; Section 2.2).
  - Training objective: predict the noise added to clean acoustic tokens (“noise‑prediction” training; [HJA20]).
  - Inference sampler: DPM‑Solver++ for fast guided sampling in ~10 steps (Section 2.2).
  - Guidance: Classifier‑Free Guidance (CFG) with scale 1.3 blends conditional (uses `hi`) and unconditional predictions for stronger alignment (Section 2.2).
  - Streaming: Because prediction happens token‑by‑token at 7.5 Hz, synthesis proceeds causally and scales to very long durations.

- Why these design choices?
  - Ultra‑low token rate (7.5 Hz) drastically shortens the sequence the LLM must handle. This is essential for hour‑long generation within a 64K token context (Introduction; Section 2.1).
  - Continuous latents + diffusion avoid discretization bottlenecks and can better capture prosody/timbre than discrete codebooks at such high compression rates (Section 2.1 and Table 3).
  - Two tokenizers separate content from acoustics, which the authors report helps long‑form generation stability (Section 2.1).

## 4. Key Insights and Innovations
- Ultra‑compressed, causal acoustic tokenizer at 7.5 Hz
  - Novelty: token rate of 7.5 frames/s (3200× compression) while maintaining high perceptual quality (Section 2.1; Table 3).
  - Why it matters: long‑form generation becomes feasible because the LLM sees far fewer acoustic tokens; the paper reports a speech‑to‑text token ratio near 2:1 (Section 1).
  - Evidence: Table 3 shows PESQ 3.068 and UTMOS 4.181 on LibriTTS test‑clean at only 7.5 tokens/s, outperforming higher‑rate systems in these metrics.

- Next‑token diffusion conditioned on LLM hidden states
  - Novelty: instead of predicting discrete acoustic tokens, the model predicts continuous VAE features with a small diffusion head per token (Section 2.2; Figure 2).
  - Why it matters: continuous targets avoid vector‑quantization errors and enable rich prosody at extreme compression, with efficient 10‑step DPM‑Solver++ sampling (Section 2.2).

- Unified long‑context, multi‑speaker sequence modeling
  - Novelty: interleave per‑speaker text and voice‑prompt features into one sequence the LLM processes (Section 2.2), rather than using separate pipelines.
  - Why it matters: simplifies architecture and allows content‑aware turn‑taking and voice consistency across up to four speakers over 90 minutes (Introduction; Figure 2).

- Empirical scaling from 1.5B to 7B LLM
  - Insight: Larger LLM improves perceptual quality (richer timbre, more natural intonation) while keeping WER competitive (Section 1; Table 1).

## 5. Experimental Analysis
- Evaluation setup
  - Long‑form podcast scenario (Section 3.1)
    - Data: 8 long conversational transcripts totaling ~1 hour; speech prompts ensure consistent timbre across models. For Gemini‑2.5‑Pro‑Preview‑TTS, default male/female voices are used because it does not accept speech prompts (Section 3.1).
    - Human evaluation: 24 annotators rate 6 models on Mean Opinion Score (MOS) for Realism, Richness, and Preference (Section 3.1).
    - Objective metrics: Word Error Rate (WER) using Whisper‑large‑v3 and Nemo ASR; speaker similarity (SIM) via WavLM‑large embeddings (Section 3.1).
  - Short‑utterance benchmarks (SEED; Section 3.2)
    - Data: ~1,000 English (CommonVoice) and ~2,000 Chinese samples (test‑en/test‑zh); metrics are WER (Whisper‑large‑v3 for English, Paraformer for Chinese) and SIM (WavLM‑large).
  - Tokenizer reconstruction (Section 3.3; Table 3)
    - Datasets: LibriTTS test‑clean/test‑other.
    - Metrics:
      - PESQ: signal‑based perceptual quality score; higher is better.
      - STOI: objective intelligibility; higher is better.
      - UTMOS: neural MOS predictor; higher is better.

- Main quantitative results
  - Long‑form conversation (Table 1)
    - Subjective MOS (higher is better):
      - VIBEVOICE‑7B: Realism 3.71 ± 0.98, Richness 3.81 ± 0.87, Preference 3.75 ± 0.94; Average 3.76 ± 0.93.
      - VIBEVOICE‑1.5B: Realism 3.59 ± 0.95, Richness 3.59 ± 1.01, Preference 3.44 ± 0.92; Average 3.54 ± 0.96.
      - Strong baselines: Gemini‑2.5‑Pro‑Preview‑TTS Average 3.66 ± 1.16; Elevenlabs v3 alpha Average 3.40 ± 1.09; others lower (Table 1).
      - Quote:
        > VIBEVOICE‑7B achieves the highest subjective scores across all three dimensions among compared systems (Table 1).
    - Objective:
      - WER (lower is better): VIBEVOICE‑1.5B 1.11 (Whisper) / 1.82 (Nemo); VIBEVOICE‑7B 1.29 / 1.95.
      - SIM (higher is better): VIBEVOICE‑7B 0.692; VIBEVOICE‑1.5B 0.548. Many open baselines have higher WER and lower SIM (Table 1).
  - Short‑utterance (SEED; Table 2)
    - VIBEVOICE‑1.5B (7.5 Hz): test‑zh CER 1.16% (SIM 0.744); test‑en WER 3.04% (SIM 0.689).
    - Some specialized short‑utterance systems achieve lower test‑en WER (e.g., CosyVoice‑2 at 2.57%), but VIBEVOICE remains competitive despite being trained for long‑form and operating at a much lower frame rate (Table 2).
  - Tokenizer reconstruction (Table 3)
    - At 7.5 tokens/s, the acoustic tokenizer attains PESQ 3.068 and UTMOS 4.181 on test‑clean and PESQ 2.848 and UTMOS 3.724 on test‑other—best among listed models on PESQ and UTMOS despite the lowest token rate. STOI is lower than some higher‑rate codecs (e.g., WavTokenizer 0.914 vs 0.828), reflecting a trade‑off between extreme compression and intelligibility proxy scores.
    - Quote:
      > “Ours (Acoustic) 1 quantizer at 7.5 tokens/s” achieves the top PESQ and UTMOS on both test‑clean and test‑other (Table 3).

- Do the experiments support the claims?
  - Long‑form conversational superiority is supported: VIBEVOICE‑7B’s MOS scores lead across all subjective dimensions in Table 1. Objective WER is also very low (≤1.95), and speaker similarity is competitive/high (0.692).
  - Short‑utterance generalization is decent but not state‑of‑the‑art on English WER; nevertheless, this is notable given the ultra‑low frame rate and long‑form focus (Table 2).
  - Tokenizer results substantiate the feasibility of 7.5 Hz compression without catastrophic loss of perceived quality (Table 3).

- Caveats and robustness
  - Subjective test set is compact (8 scripts totaling ~1 hour) and uses 24 annotators (Section 3.1). While thorough for long audio (~6 hours of listening per annotator), broader genres and languages are not included.
  - One baseline (Gemini) could not use voice prompts, potentially underrepresenting its speaker‑matching ability (Section 3.1).
  - No ablations isolating the impact of the semantic tokenizer, CFG scale, sampler steps, or curriculum are reported. The 1.5B vs 7B comparison indicates scaling helps, but the specific contribution of each design choice remains unquantified.

## 6. Limitations and Trade-offs
- Stated limitations (Section 4)
  - Language coverage: English and Chinese only. Other languages may produce “unexpected audio outputs.”
  - Non‑speech audio: background sounds, music, and sound effects are not modeled.
  - Overlapping speech: simultaneous multi‑speaker overlap is not explicitly handled; turns are sequential.
  - Responsible use: potential misuse for impersonation and disinformation; the model is released for research only.

- Architectural and computational trade‑offs
  - While 7.5 Hz greatly reduces sequence length, STOI drops compared to some higher‑rate tokenizers (Table 3). This implies a trade‑off: extreme compression favors scalability but can impact intelligibility proxies.
  - Inference still runs an iterative diffusion process per acoustic token; although sampling is only ~10 steps, very long outputs still accumulate latency.
  - The system relies on a sizable LLM (up to 7B) and a long context window (up to 65K tokens during training), implying significant training/inference memory requirements for long dialogues.

- Evaluation coverage
  - Long‑form tests involve curated scripts and controlled voice prompts; spontaneous conversations, noisy environments, and code‑switching are not evaluated.
  - Objective intelligibility and alignment are measured via ASR WER, which can confound evaluation if ASR itself is biased by speaking style or prosody.

- Open questions
  - How much do the semantic tokenizer and hybrid conditioning improve long‑form stability versus acoustic‑only conditioning?
  - What is the failure behavior at or beyond the 90‑minute mark (e.g., drift, voice leakage across speakers)?
  - How sensitive is quality to CFG scale, number of diffusion steps, and speaker‑turn density?

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that extreme acoustic compression (7.5 Hz) plus token‑level diffusion can carry long‑form, multi‑speaker generation without sacrificing perceptual quality (Figure 1; Table 3). This reframes long‑form TTS as a tractable long‑context sequence modeling problem rather than a stitching/concatenation problem.
  - Suggests continuous latents predicted via diffusion are a strong alternative to discrete acoustic tokens for large‑context speech LMs.

- Follow‑up research enabled or suggested
  - Overlap modeling: introduce mechanisms for controlled, simultaneous speech (e.g., multi‑stream diffusion or mask‑based generation).
  - Cross‑lingual and code‑switching: extend tokenizers and training data to additional languages; test robustness in multilingual dialogues.
  - Ablations and interpretability: quantify the contributions of the semantic tokenizer, CFG, and sampler; analyze how the LLM manages speaker identity and turn‑taking.
  - Efficiency: explore distillation or fewer diffusion steps; test non‑iterative decoders conditioned on LLM states while preserving long‑form stability.
  - Safety and watermarking: develop built‑in safeguards (e.g., watermarking in acoustic latents) to mitigate misuse.

- Practical applications
  - Scripted podcast and audio drama production with consistent voices and controlled turn‑taking.
  - Multi‑narrator audiobooks and educational content.
  - Dialogue prototyping for games and virtual agents.
  - Given current limitations, deployments should avoid background sound requirements, overlapping speech, and unsupported languages, and include safety filters and human review (Section 4).

Quote highlights anchoring claims:
- “VIBEVOICE can synthesize long‑form speech for up to 90 minutes (in a 64K context window length) with a maximum of 4 speakers” (Introduction; Figure 2).
- “We… developed a causal speech tokenizer that achieves a 3200× compression rate (i.e., 7.5 Hz frame rate)… [with] a speech‑to‑text token ratio of approximately 2:1” (Introduction; Section 2.1).
- “VIBEVOICE‑7B… 3.71 Realism, 3.81 Richness, 3.75 Preference… outperforming strong open/closed‑source systems” (Table 1; Figure 1).
- “Ours (Acoustic) … 7.5 tokens/s … PESQ 3.068, UTMOS 4.181 (test‑clean)” (Table 3).
