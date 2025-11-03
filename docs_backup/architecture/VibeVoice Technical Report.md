# VibeVoice Technical Report

**ArXiv:** [2508.19205](https://arxiv.org/abs/2508.19205)
**Authors:** Zhiliang Peng, Jianwei Yu, Wenhui Wang, Yaoyao Chang, Yutao Sun, Li Dong, Yi Zhu, Weijiang Xu, Hangbo Bao, Zehua Wang, Shaohan Huang, Yan Xia, Furu Wei
**Institutions:** Microsoft Research

## 🎯 Pitch

VIBEVOICE revolutionizes long-form, multi-speaker text-to-speech synthesis by integrating a groundbreaking ultra-low-rate continuous speech tokenizer with an LLM-conditioned next-token diffusion, compressing audio into just 7.5 tokens per second. This innovation enables the generation of up to 90 minutes of coherent multi-speaker dialogue while maintaining high perceptual quality, significantly advancing the creation of natural conversational audio for podcasts and audiobooks.

---

## 1. Executive Summary
VIBEVOICE introduces a long-form, multi-speaker text-to-speech (TTS) system that scales to 90 minutes of coherent dialogue by combining an ultra-low-rate continuous speech tokenizer with a next-token diffusion generator conditioned on a large language model (LLM). The key significance is practical scalability: by compressing audio to 7.5 tokens per second (≈3200× compression), the system keeps entire multi-speaker conversations within a 64K context window while maintaining high perceptual quality and outperforming strong open/closed-source baselines in subjective tests (Figure 1, Table 1).

## 2. Context and Motivation
- Problem addressed
  - Generating natural, long-form, multi-speaker conversation audio (e.g., podcasts, multi-voice audiobooks) that preserves turn-taking, expressiveness, and consistency over tens of minutes. Traditional TTS can concatenate short utterances, but this breaks conversational rhythm and context continuity.
- Why it matters
  - Real-world impact: Enables production-quality spoken content at podcast/audiobook scale with multiple voices and long-range context.
  - Technical significance: Demonstrates how to keep long audio within manageable sequence lengths and how to condition speech synthesis on rich, conversation-level context.
- Prior approaches and gaps
  - Existing work excels at short, single-speaker speech (cited in Section 1: NaturalSpeech-2, Voicebox, SEED-TTS, etc.), but struggles with long, multi-speaker generation.
  - Recent long-conversation systems (e.g., MoonCast, SesameAILabs-CSM, Nari Labs Dia; references in Section 1) either are not open-source or show instability and limited duration (Section 1, paragraph 2).
  - Technical bottleneck: audio tokens come at high frame rates (often 25–50 Hz), so modeling many minutes of audio explodes sequence length.
- How VIBEVOICE positions itself
  - It introduces a causal, continuous acoustic tokenizer operating at 7.5 Hz (≈3200× compression; Sections 1 and 2.1) and couples it to an LLM-driven next-token diffusion generator (Section 2.2). This combination keeps context lengths feasible (64K tokens ≈ 90 minutes; Figure 2 caption and Section 1) while retaining audio fidelity.

## 3. Technical Approach
VIBEVOICE consists of (i) two speech tokenizers (acoustic and semantic), (ii) an LLM backbone, and (iii) a token-level diffusion head that predicts acoustic features token by token. Figure 2 summarizes the overall architecture and generation flow.

- Two speech tokenizers (Section 2.1)
  - Acoustic tokenizer (continuous, causal σ-VAE)
    - Goal: compress waveform into continuous low-rate latent vectors that preserve perceptual quality.
    - σ-VAE concept: instead of learning the latent variance per sample (as in standard VAE), it fixes variance by sampling σ from a predefined distribution N(0, Cσ) to avoid variance collapse when used with autoregressive modeling. A latent `z` is sampled as `z = μ + σ ⊙ ε`, with `ε ~ N(0, 1)` (Section 2.1).
    - Architecture: 7-stage, mirror-symmetric encoder–decoder with hierarchical modified Transformer blocks that replace self-attention with 1D depth-wise causal convolutions (streaming-friendly). Six downsampling layers give 3200× compression from 24 kHz, yielding 7.5 tokens/second (Section 2.1). Each of encoder and decoder is ~340M parameters. Training objective follows DAC (a GAN-based neural codec training scheme; Section 2.1).
    - Why this matters: 7.5 Hz drastically reduces sequence length versus 25–50 Hz codecs, making 64K-token contexts viable for hour-scale generation (Section 1).
  - Semantic tokenizer (deterministic, ASR-supervised)
    - Goal: extract content-centric features aligned with text.
    - Architecture mirrors the acoustic encoder but without VAE; trained by attaching a Transformer decoder to predict transcripts (ASR proxy task), then discarding the decoder after pretraining (Section 2.1).
    - Rationale: separating content (semantic) from acoustics improves long-form generation stability (Section 2.1 opening sentence).

- Input representation and conditioning (Section 2.2)
  - Inputs combine user-provided voice prompts (speaker “voice fonts”) and text scripts. For `N` speakers:
    - `[Speaker1: z1, …, SpeakerN: zN] + [Speaker1: T1, …, SpeakerN: TN]`
    - `zN` are acoustic latent representations (voice prompts), and `TN` are the text scripts per role (Section 2.2).
  - The model also encodes already generated speech segments using both tokenizers to form hybrid context features for auto-regressive modeling (Section 2.2).

- LLM backbone and token-level diffusion head (Section 2.2)
  - LLM: Qwen2.5 in 1.5B and 7B parameter sizes (Section 2.2).
  - The LLM processes the hybrid context (speaker IDs, voice prompts, scripts, previous segment features) and produces per-token hidden states `h_i`.
  - Diffusion head: a lightweight 4-layer module that predicts the noise added to clean acoustic VAE features `z_a,i` during training (a standard diffusion training target). At inference, it iteratively denoises Gaussian noise to generate `z_a,i`, conditioned on `h_i` (Section 2.2).
  - Guidance and sampling:
    - Uses Classifier-Free Guidance (CFG) to combine conditional and unconditional predictions; guidance scale is 1.3 (Section 2.2).
    - Uses DPM-Solver++ for efficient sampling; only 10 denoising steps (Section 2.2).
  - Training and scaling strategy:
    - The acoustic and semantic tokenizers are frozen; only the LLM and diffusion head are trained (Section 2.2).
    - Curriculum on input sequence length: increase from 4,096 to 65,536 tokens to teach the model long contexts (Section 2.2).
  - Why next-token diffusion?
    - “Next-token diffusion” generates continuous tokens auto-regressively with a diffusion step at each token, unifying continuous outputs with token-by-token LLM conditioning (Figure 2 and references to LatentLM in Sections 1 and 2.2). This allows streaming synthesis: each token’s acoustic latent is refined before moving to the next.

- End-to-end generation flow (Figure 2; Sections 1 and 2.2)
  1. User provides voice prompts for up to 4 speakers and a multi-speaker script.
  2. The system constructs the interleaved sequence of speaker IDs, voice latents, and text.
  3. The LLM processes this sequence and produces hidden states token by token.
  4. The diffusion head converts each hidden state into an acoustic VAE latent by iterative denoising.
  5. The acoustic decoder reconstructs the waveform segment. The process continues streaming for up to 90 minutes within a 64K-token window.

## 4. Key Insights and Innovations
- Ultra-low-rate continuous acoustic tokenizer (fundamental)
  - What’s new: A causal σ-VAE tokenizer that runs at 7.5 Hz (≈3200× compression) while retaining high perceptual quality (Sections 1 and 2.1).
  - Why it matters: This shifts the speech-to-text token ratio to ≈2:1 (two speech tokens per BPE text token; Section 1), enabling hour-scale sequences within a 64K context window (≈90 min; Figure 2 caption and Section 1). Table 3 shows state-of-the-art PESQ and UTMOS at a fraction of the token rate of prior codecs.
- Token-level next-token diffusion conditioned by an LLM (fundamental)
  - What’s new: Generates continuous acoustic tokens auto-regressively via diffusion, with the LLM’s per-token hidden states guiding the denoising (Section 2.2, Figure 2).
  - Why it matters: Harmonizes language-level planning (turn-taking, semantics, style) with fine-grained acoustic synthesis, enabling natural conversational flow over long contexts.
- Hybrid speech representation (incremental but impactful)
  - What’s new: Separate acoustic (σ-VAE) and semantic (ASR-aligned) tokenizers, both used as context features during generation (Section 2.1 and 2.2).
  - Why it matters: Helps stabilize long-form synthesis and preserve both content and timbre across speakers and turns.
- Simplified conditioning design (incremental)
  - What’s new: Removes extra prior modules by concatenating voice latents and text scripts directly into the LLM’s sequence (Section 1).
  - Why it matters: A cleaner architecture reduces engineering overhead and eases scaling from 1.5B to 7B parameters with noticeable perceptual gains (Section 1; Table 1).

## 5. Experimental Analysis
- Evaluation methodology (Section 3)
  - Long-form conversational benchmark (“VIBEVOICE Podcast”; Section 3.1)
    - Data: 8 long conversational transcripts totaling ≈1 hour (Section 3.1).
    - Controls: Consistent voice prompts across models; Gemini TTS lacks speech-prompt control, so its default voices are used (Section 3.1).
    - Metrics:
      - Objective: Word Error Rate (WER) via Whisper-large-v3 and Nemo ASR; Speaker similarity (SIM) via WavLM-large embeddings (Section 3.1).
      - Subjective (MOS, 1–5 scale): 24 annotators rate Realism, Richness, and overall Preference (Section 3.1).
    - Baselines: Nari Labs Dia, Mooncast, SesameAILabs-CSM, Higgs Audio V2, Elevenlabs v3 alpha, Gemini 2.5 Pro preview TTS (Table 1 and Section 3.1).
  - Short-utterance benchmark (SEED test sets; Section 3.2)
    - Data: CommonVoice (≈1,000 English samples `test-en`; ≈2,000 Chinese samples `test-zh`).
    - Metrics: WER (Whisper-large-v3) for English; CER (Paraformer) for Chinese; SIM via WavLM-large (Section 3.2; Table 2).
  - Tokenizer reconstruction (Section 3.3)
    - Data: LibriTTS `test-clean` and `test-other`.
    - Metrics: PESQ (perceptual speech quality), STOI (intelligibility), UTMOS (a MOS predictor) (Section 3.3; Table 3).

- Main quantitative results
  - Long-form conversation (Table 1; Figure 1)
    - Subjective MOS:
      - `VIBEVOICE-7B`: Realism 3.71±0.98, Richness 3.81±0.87, Preference 3.75±0.94; Average 3.76±0.93.
      - `VIBEVOICE-1.5B`: Realism 3.59±0.95, Richness 3.59±1.01, Preference 3.44±0.92; Average 3.54±0.96.
      - Strong baselines: Gemini 2.5 Pro preview TTS achieves Average ≈3.66; Elevenlabs v3 alpha ≈3.40 (Table 1). The figure’s bar chart (Figure 1) shows the same relative ordering.
    - Objective:
      - WER (lower is better): `VIBEVOICE-7B` achieves 1.29 (Whisper) and 1.95 (Nemo); `VIBEVOICE-1.5B` achieves 1.11 and 1.82; Gemini 2.5 Pro preview TTS has 1.73 and 2.43 (Table 1).
      - SIM (higher is better): `VIBEVOICE-7B` 0.692 vs `VIBEVOICE-1.5B` 0.548 (Table 1).
    - Takeaways:
      - The 7B model clearly improves perceptual quality and similarity over the 1.5B model (Table 1).
      - The 1.5B model has slightly lower WER but lower SIM and subjective MOS; the 7B trades a modest WER increase for better timbre/expressiveness (Table 1).
  - Short utterances (Table 2)
    - Despite being trained for long-form, `VIBEVOICE-1.5B` is competitive:
      - Chinese `test-zh`: CER 1.16%, SIM 0.744 (strong results compared with MaskGCT 2.27% and CosyVoice 2 at 1.45% CER).
      - English `test-en`: WER 3.04%, SIM 0.689 (not the best WER—Spark TTS reports 1.98%—but competitive, especially considering VIBEVOICE’s 7.5 Hz).
    - Efficiency note: VIBEVOICE’s 7.5 Hz frame rate substantially reduces decoding steps per second of audio versus others at 25–50 Hz (Table 2).
  - Tokenizer reconstruction (Table 3; Section 3.3)
    - At just 7.5 tokens/s, the acoustic tokenizer achieves:
      - `test-clean`: PESQ 3.068, UTMOS 4.181 (best in table); STOI 0.828.
      - `test-other`: PESQ 2.848, UTMOS 3.724 (best in table); STOI 0.823.
    - Comparison: WavTokenizer at 75 tokens/s scores PESQ 2.373, UTMOS 4.049. Encodec at 600 tokens/s reaches PESQ 2.72. This underscores the tokenizer’s efficiency/quality trade-off advantage.

- Support for claims and caveats
  - The claim of long-duration capability is documented:
    - “VIBEVOICE can synthesize long-form speech for up to 90 minutes (in a 64K context window) with up to 4 speakers” (Section 1 and Figure 2 caption).
    - Figure 1 highlights “5,000+ seconds of audio” and consistent subjective gains.
  - The long-form evaluation set is compact (8 transcripts, ~1 hour) but deeply annotated (≈6 hours of listening per annotator; Section 3.1), which is credible yet may not capture all conversational regimes (e.g., debates, interruptions).
  - No ablation studies are presented on key choices (e.g., the exact contribution of semantic tokenizer, the effect of CFG scale, or the number of diffusion steps). Robustness to prompt style, accents, or code-switching is not reported.

## 6. Limitations and Trade-offs
- Assumptions and scope (Section 4)
  - Language coverage: “English and Chinese only.” Other languages can produce “unexpected audio outputs.”
  - Sound domain: Speech only—no background noise, music, or sound effects.
  - Conversational structure: “Does not explicitly model or generate overlapping speech segments,” so cross-talk is unsupported.
- Data and evaluation coverage
  - Long-form subjective set is relatively small (8 transcripts) and English/Chinese only (Section 3.1).
  - Fairness note: Gemini TTS lacks voice-prompt control and thus uses default voices (Section 3.1), which can confound speaker-similarity comparisons.
- Architectural trade-offs
  - LLM-in-the-loop: Using a 7B LLM improves MOS and SIM but modestly worsens WER versus 1.5B (Table 1), and will increase memory/compute.
  - Diffusion steps: Although only 10 steps are used (efficiency win; Section 2.2), diffusion remains iterative; real-time factors are not reported.
  - Tokenizers are large (≈340M parameters each for encoder/decoder; Section 2.1) but are frozen during training; still, end-to-end memory footprint may be significant.
- Open questions
  - How does performance change with different frame rates or alternative tokenizers (e.g., discrete RVQ-based codecs)?
  - How stable is generation beyond 90 minutes, or with more than 4 speakers?
  - Robustness to noisy prompts, code-switching, and diverse recording conditions is not evaluated.
- Risk considerations (Section 4)
  - The report explicitly warns about deepfake risks and advises research-only use until further testing.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that long-form conversational TTS can be scaled with an LLM by aggressively reducing acoustic token rate without sacrificing perceptual quality (Figure 1; Table 3). This reframes long-context speech generation as a sequence modeling problem tractable within standard LLM context limits (64K).
- What this enables next
  - Richer dialogue agents: With multi-speaker conditioning and long-range memory, conversational agents can narrate, host, and role-play over podcast-length sessions.
  - Cross-lingual style transfer: The report notes enhanced transfer capabilities, including cross-lingual cases when scaling to 7B (Section 1). Systematic multilingual extensions look promising.
  - Research on overlapping speech: Adding explicit overlap modeling (e.g., diarization-aware or mixture-aware latents) could enable realistic interruptions and cross-talk.
  - Ablations and controls: Studies on the individual contributions of semantic tokens, CFG strength, number of diffusion steps, and alternative samplers would clarify design trade-offs.
  - Safety/verification: Integrations with watermarking, speaker-consent verification, or content authenticity checks are natural follow-ups given the stated risks (Section 4).
- Practical applications
  - Multi-voice audiobooks, dramatized podcasts, educational dialogues, scripted radio plays, and dubbing with consistent character “vibes” over long durations.
  - Production considerations: Until broader language and overlap support are added and more robustness tests are done, deployment should remain controlled, aligning with the paper’s “research use only” guidance (Section 4).

> Core contribution in one line: an ultra-low-rate continuous speech tokenizer plus LLM-conditioned next-token diffusion makes 90-minute, multi-speaker conversational TTS feasible in a 64K-token window with strong subjective quality (Sections 1–2; Figure 1; Table 1; Table 3).
