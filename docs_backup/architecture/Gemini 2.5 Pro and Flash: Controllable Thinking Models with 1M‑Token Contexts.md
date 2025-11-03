# Gemini 2.5 Pro and Flash: Controllable Thinking Models with 1M‑Token Contexts

**ArXiv:** [2506.XXXXX](https://arxiv.org/abs/2506.XXXXX)
**Authors:** Shrestha Basu Mallick, Logan Kilpatrick, Tulsee Doshi
**Institutions:** Google DeepMind

## 🎯 Pitch

The Gemini 2.X model family introduces a groundbreaking 'Thinking' capability that allows for dynamic reasoning with a controllable budget, enabling unprecedented performance across coding, reasoning, and multimodal tasks with 1M+ token contexts. This advancement is crucial as it addresses the balance between capability and cost, fostering real-world applications in search, education, and agentic systems, while demonstrating enhanced safety and reliability standards necessary for high-impact deployments.

---

## 1. Executive Summary (2–3 sentences)
This report introduces the Gemini 2.X model family—centered on `gemini-2.5-pro` (“Pro”) and `gemini-2.5-flash` (“Flash”)—and demonstrates state‑of‑the‑art performance on difficult coding, math/reasoning, factuality, and multimodal (audio/video) tasks, with long‑context processing (>1M tokens) and integrated tool use. The core advance is a “Thinking” capability: the model can allocate additional inference‑time compute with a controllable budget to reason more deeply, enabling new “agentic” workflows (multi‑step, tool‑using systems) while improving helpfulness and safety metrics (Sections 2.5, 2.6, 3; Tables 3–6, 7–9).

## 2. Context and Motivation
- Problem/gap addressed
  - Existing large models often trade off cost, latency, and capability; struggle with integrated long‑context + multimodal reasoning; and lack controllable inference‑time reasoning (“thinking”) that scales accuracy with compute. This limits real‑world “agentic” systems that must browse, code, or reason across long documents and hours of video (Sections 1, 2.5, 2.6).
- Importance
  - Practical: Search, coding agents, education, and creative tools need faithful long‑context, grounded answers with sources and tools; enterprises need predictable cost/latency (Table 1; Figure 2).
  - Scientific: Probing whether reasoning accuracy can reliably scale with inference compute, not just with model size; evaluating safety as capabilities grow (Sections 2.5, 5, 5.7; Tables 7–10).
- Prior approaches and their limits
  - Earlier Gemini 1.5 models had long context but weaker reasoning and no controllable thinking; many competitive models either do not support 1M+ context or lack native multimodality/tool use at similar levels (Table 4; Section 2.6 Long Context).
- Positioning
  - Gemini 2.5 targets the “Pareto frontier of model capability vs cost” (Figure 1), delivering:
    - long context + multimodality + tool use as built‑ins (Table 1),
    - controllable Thinking across domains (Figures 3–4),
    - and competitive or state‑of‑the‑art (SoTA) scores across coding, reasoning, factuality, audio, and video (Tables 3–6).
  - It also presents systematic safety evaluation and infrastructure advances to train at unprecedented scale (Sections 2.3, 5; Tables 7–10).

## 3. Technical Approach
- Model family and architecture
  - Pro and Flash are sparse `Mixture‑of‑Experts (MoE)` transformers (a routing mechanism that activates only a subset of experts per token to decouple total capacity from per‑token cost) with native multimodal encoders for text, vision, and audio (Section 2.1).
  - Long context: both models process sequences up to 1M tokens (Table 1), including codebases and multi‑hour audio/video; video uses an improved visual tokenization (66 tokens/frame) to pack about 3 hours within the context window (Section 2.6 Video).
  - Distillation for smaller variants reduces serving cost by approximating the teacher’s distribution with a k‑sparse target to lower storage/throughput overhead while preserving quality (Section 2.1).
- Data and post‑training
  - Pre‑training uses large, diverse text/code/image/audio/video mixtures with improved filtering/dedup; cutoffs are June 2024 for 2.0 and January 2025 for 2.5 (Section 2.2).
  - Post‑training increases compute for reinforcement learning (RL), emphasizes verifiable/model‑based rewards (so the model is trained to check and justify), and integrates tool use (Section 2.4).
- “Thinking”: controllable inference‑time reasoning
  - Concept: Train with RL so the model can spend extra inference tokens on internal computation (a scratchpad) before answering. A user‑provided `thinking budget` limits this cost/latency (Section 2.5).
  - Mechanism: The model interleaves internal thought with tool use (e.g., search, code execution) and can dynamically decide how long to think (Figures 3–4). Benchmarks show accuracy scaling with budget on AIME 2025, LiveCodeBench, and GPQA (Figure 4).
- Safety‑aware post‑training
  - Uses supervised fine‑tuning (SFT), data reward models (DRMs) trained on human preferences, and “Critic” graders to reinforce helpfulness while minimizing policy violations (Section 5.3).
  - Automated Red Teaming (ART) employs attacker/critic agents to find violations and improve data/metrics (Section 5.5).
- Training infrastructure (for scale and reliability)
  - First training on TPUv5p across multiple pods; two key reliability techniques:
    - `Slice‑Granularity Elasticity`: continue training while a slice recovers to reduce downtime (Section 2.3).
    - `Split‑Phase SDC Detection`: fast deterministic replays and per‑device checksums to localize hardware‑induced Silent Data Corruption (SDC), with ~0.25% steps replayed and ~6% true SDC among replays (Section 2.3).
  - Single controller Pathways orchestration simplifies fault tolerance at scale (Section 2.3).
- Capability‑specific improvements (high level)
  - Code: more and better code data, task‑like post‑training, evaluation scaffolds for repo‑level work; improved IDE/agent use cases (Section 2.6 Code).
  - Factuality: native search tool use and long‑context grounding (FACTS), with strong short‑form factuality (Section 2.6 Factuality).
  - Audio: adds low‑latency streaming dialog and controllable TTS; integrates thinking and tools in audio modality (Section 2.6 Audio; Section 2.7 “2.5 Audio Generation”).
  - Video: higher temporal understanding; 3‑hour videos; can convert demos to interactive coding apps (Section 2.6 Video; Baddepudi et al., 2025 referenced in text).
  - Agents: shows real‑world agentic systems like Gemini Deep Research and “Gemini Plays Pokémon” (Sections 2.6 Deep Research; 4.1; Appendix 8.2).

## 4. Key Insights and Innovations
- Controllable Thinking as a first‑class, cross‑modal capability
  - What’s new: A single model that can spend and scale inference‑time compute (budgeted “thinking tokens”) across text, image, audio, and video tasks, with measurable gains (Figures 3–4).
  - Why it matters: It decouples answer quality from fixed latency, letting users trade cost for accuracy and enabling robust multi‑step agents.
- Long‑context multimodality at practical scale
  - What’s new: Stable >1M‑token processing combined with 3‑hour video ingestion via efficient visual tokens (Section 2.6 Video), and demonstrated retrieval/recall over 46‑minute videos (Appendix 8.5; Table 12).
  - Why it matters: Enables whole‑codebase editing, hour‑long video Q&A, and evidence‑grounded synthesis across many sources; SoTA on several long‑context tasks (Table 3).
- End‑to‑end agentic workflow design
  - What’s new: Native tool use + search + code execution tightly integrated with Thinking and long context; “Deep Research” achieves 26.9%–32.4% on Humanity’s Last Exam (HLE) with higher compute (Section 2.6 Deep Research).
  - Why it matters: Moves from single‑turn Q&A to reliable multi‑turn task completion (web research, repo‑level fixes, app generation).
- Safety and reliability engineering at scale
  - What’s new: End‑to‑end safety pipeline (ART, FSF evaluations, memorization audits), targeted defenses against indirect prompt injection with measured Attack Success Rate (ASR) drops (Table 9), and training‑time hardware fault tolerance (Section 2.3; Section 5).
  - Why it matters: Capability gains paired with safety/robustness evidence is required for deployment in high‑impact products (Section 4.3).

## 5. Experimental Analysis
- Evaluation methodology
  - Gemini scores use the AI Studio API with pass@1 and default sampling; “single attempt” unless “multiple attempts” is noted (Section 3.1; Table 2 for model IDs).
  - Competitive results are mostly provider‑reported or from public leaderboards; SWE‑bench Verified uses differing scaffolds across providers, so those numbers are not strictly comparable (Section 3.1).
- Benchmarks and metrics (selected)
  - Coding: LiveCodeBench, Aider Polyglot, SWE‑bench Verified (Table 3).
  - Reasoning/Math: AIME 2025, GPQA (diamond) (Table 3).
  - Factuality: SimpleQA (parametric knowledge, F1), FACTS Grounding (faithfulness) (Table 3).
  - Long‑context: LOFT (retrieval), MRCR‑V2 (multi‑needle long‑context reasoning) at ≤128k and exact 1M (Table 3).
  - Vision: MMMU, Vibe‑Eval (Reka), ZeroBench, BetterChartQA (Table 3).
  - Audio: FLEURS (WER↓), CoVoST2 (BLEU↑) (Table 5).
  - Video: ActivityNet‑QA, LVBench, VideoMME, 1H‑VideoQA, QVHighlights, Minerva, Neptune (Table 6).
  - Safety/Robustness: Automated Red Teaming metrics (Tables 7–8), Indirect Prompt Injection ASR (Table 9), Memorization (Figure 8), FSF Critical Capability Levels (Table 10).
- Main quantitative results (highlights)
  - Coding (Table 3):
    - LiveCodeBench: `2.5 Pro 74.2%` vs `1.5 Pro 30.3%`.
    - Aider Polyglot: `2.5 Pro 82.2%` vs `1.5 Pro 16.9%`.
    - SWE‑bench Verified: single attempt `2.5 Pro 59.6%`; multiple attempts `67.2%`.
  - Reasoning/Math (Table 3): AIME 2025 `2.5 Pro 88.0%` (vs `1.5 Pro 17.5%`), GPQA diamond `86.4%` (vs `58.1%`).
  - Factuality (Table 3): SimpleQA `2.5 Pro 54.0%`, FACTS Grounding `87.8%`.
  - Long‑context (Table 3):
    - LOFT hard ≤128k: `87.0%`; LOFT 1M: `69.8%`.
    - MRCR‑V2 ≤128k: `58.0%`; MRCR‑V2 1M: `16.4%` (good ≤128k, still challenging at full 1M).
  - Vision (Table 3): MMMU `82.0%`, Vibe‑Eval `67.2%`, ZeroBench `4.5%` (improved but still very hard), BetterChartQA `72.4%`.
  - Versus other models (Table 4):
    - Aider Polyglot: `2.5 Pro 82.2%` vs `o3‑high 79.6%`, `o4‑mini 72.0%`, `Claude 4 Sonnet (Extended Thinking) 72.0%`.
    - GPQA diamond single attempt: `2.5 Pro 86.4%` vs `o3‑high 83.3%`, `Grok 3 Beta 81.0%`.
    - HLE (no tools): `2.5 Pro 21.6%` vs `o3‑high 20.3%`, `o4‑mini 18.1%` (Gemini Deep Research with higher compute reports 26.9%–32.4% in Section 2.6).
    - Long‑context LOFT ≤128k: `2.5 Pro 87.0%`—highest among reported; and unique support for 1M+.
  - Audio (Table 5): FLEURS WER `6.66` (lower is better) and CoVoST2 BLEU `38.48`—competitive/SoTA under matched prompts/inputs against GPT‑4o variants.
  - Video (Table 6): SoTA/competitive under matched prompts:
    - 1H‑VideoQA: `81.0` vs GPT‑4.1 `56.8`.
    - VideoMME (audio+visual+subtitles): `86.9` vs GPT‑4.1 `79.6`.
    - VideoMMMU: `83.6` vs GPT‑4.1 `60.9`.
- Performance–cost profile
  - Throughput: Figure 2 shows very high output tokens/sec for `2.5 Flash` and strong throughput for `2.5 Pro`, enabling cost/latency selection.
  - Thinking ablations: Figures 3–4 show accuracy rising monotonically with thinking budget for AIME, LiveCodeBench, and GPQA.
- Safety and robustness
  - Helpfulness and policy metrics improved relative to earlier models:
    > Table 7: “Helpfulness/Instruction Following” ↑ for `2.5 Pro` vs `1.5 Pro` by +14.8 points, with English/i18n policy violations generally ↓ (small ↑ for image‑to‑text, flagged as non‑egregious).
  - Automated Red Teaming (ART) trends:
    > Table 8: “Dangerous Content policy violations” reduce from `43.5% (1.5 Pro)` to `24.3% (2.5 Pro)`; “Helpfulness violations” from `8.9%` to `6.1%`.
  - Indirect Prompt Injection defenses:
    > Table 9: Best ASR reductions against Actor‑Critic and Beam Search; TAP ASR reduces from `64.8% (1.5 Flash)` to `30.8% (2.5 Pro)` (lower is better), though `2.5 Flash` is most resilient overall.
  - Memorization:
    > Figure 8 (Left): Gemini 2.X shows markedly lower total memorization than prior models; Figure 8 (Right): “No instances of personal information” in text classified as memorization for Gemini 2.X under the reported detection thresholds.
  - Frontier Safety Framework (FSF):
    > Table 10: No Critical Capability Levels (CCLs) reached across CBRN, Cybersecurity, ML R&D, or Deceptive Alignment. A cyber “Uplift Level 1” alert threshold is triggered (more frequent testing/mitigations planned).
  - Cyber capability probes:
    > Figure 11 (Key skills): `2.5 Pro` solves `6/12` hard challenges—approaching experienced professional level on select tasks; `InterCode-CTF` is saturated, `Hack the Box` still too hard (Figure 10).
- Qualitative agentic demo (“Gemini Plays Pokémon”)
  - Long‑horizon competence: completed the game; with a fixed harness, the second run halved time to `406.5` hours (Section 4.1; Figure 6). Specialized sub‑agents (`pathfinder`, `boulder_puzzle_strategist`) solved long sequences/mazes with contexts >100k tokens (Appendix 8.2; Figure 14).
  - Observed failure modes: reliance on RAM‑derived text over raw pixels (“screen reading”), fixation/looping in very long contexts, and “topological traps” in maze reasoning (Appendix 8.2).

Overall assessment: The breadth of benchmarks, explicit thinking‑budget ablations, and multiple safety analyses substantiate the central claims. Some cross‑model comparisons (e.g., SWE‑bench scaffolds) are not perfectly apples‑to‑apples (Section 3.1), and full‑window 1M‑token generative reasoning still lags retrieval, but the evidence is strong that Gemini 2.5 advances capability and reliability at useful cost/latency points.

## 6. Limitations and Trade-offs
- Long‑context generation remains challenging
  - Despite strong ≤128k results, MRCR‑V2 at 1M is `16.4%` (Table 3). The Pokémon case study notes looping/goal fixation as the context grows “significantly beyond 100k” tokens, hinting at open problems in long‑horizon planning over massive traces (Appendix 8.2).
- Comparability and evaluation caveats
  - Provider‑reported baselines and differing scaffolds (e.g., SWE‑bench Verified) limit strict cross‑model comparability (Section 3.1). Some leaderboards evolve over time; reported windows matter (e.g., LiveCodeBench dates).
- Cost/latency vs accuracy
  - Thinking improves accuracy but increases tokens and latency (Figures 3–4). While budgets control this, practical deployments must tune for SLA and cost targets.
- Multimodal output availability
  - Some output modalities (e.g., audio/image generation) are marked “Preview/Experimental” in Table 1, indicating partial availability and ongoing productization.
- Security posture is improving but not solved
  - Although ASRs drop substantially, Pro remains less resilient than Flash on some attack methods (Table 9). FSF cyber “Uplift Level 1” hits an alert threshold, signaling rapid capability growth and the need for accelerated mitigations (Table 10).
- Data/engineering opacity
  - The report summarizes data curation and safety pipelines but does not disclose exhaustive dataset composition, exact training compute, or full ablation details, which can hinder reproducibility.

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes controllable Thinking as a practical, cross‑modal lever, not a research novelty—users can “buy” accuracy with inference compute. Combining this with 1M+ context and 3‑hour video puts whole‑repository, lecture‑length, and multi‑document tasks in reach for mainstream agents (Tables 1, 3, 6; Figures 3–4).
  - Demonstrates that capability can scale with safety: lowered memorization and prompt‑injection ASRs, plus FSF governance, enable broader deployment in consumer and enterprise products (Tables 7–10; Figure 8).
- Research directions enabled
  - Long‑horizon agent design for million‑token settings: memory structures, retrieval‑augmented planning, and anti‑looping policies that distinguish retrieval from generative reasoning at scale (Appendix 8.2 observations; MRCR‑V2 1M in Table 3).
  - Compute‑aware orchestration: adaptive thinking budgets tied to task difficulty, tool confidence, and cost constraints; theoretical models of budget‑accuracy trade‑offs.
  - Safety under increasing autonomy: richer ART ecosystems, stronger defenses for indirect prompt injection across toolchains, and more granular FSF thresholds/models for “Uplift.”
  - Evaluation infrastructure: harder, contamination‑resistant, economically meaningful benchmarks for code agents, research assistants, and long‑video understanding (Section 6 Discussion).
- Practical applications
  - High‑leverage coding agents (repo‑level fixes, web/app generation) with measurable gains (Table 3); research assistants with search + reasoning (Deep Research; Section 2.6).
  - Education and content: generate interactive learning apps from lectures; faithful long‑document summarization with sources (Tables 6; Section 4.2).
  - Multimodal assistants: low‑latency native audio dialog, controllable TTS, advanced video Q&A for media analysis and support operations (Sections 2.6 Audio/Video; Tables 5–6).

> In short, Table 3 and Table 6 document broad SoTA‑level capability; Figures 3–4 show a controllable path to more accuracy; and Tables 7–10 evidence a maturing safety stack. Together, these enable more reliable, cost‑aware agents that reason across text, code, audio, and hours‑long video at production scale.
