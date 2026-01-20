# Paper Summaries Knowledge Base

Visit the site directly.

https://summary-of-some-paper-in-cuda.readthedocs.io/en/latest/llm-systems/2512.02556-DeepSeek-V3.2-PushingtheFrontierofOpenLargeLanguageModels/?h=2512#342-deepseek-sparse-attention-dsa-what-it-is-and-how-it-works

## 📂 Categories

| Category | Papers | Tokens | Avg/Paper |
|----------|--------|--------|-----------|
| **Pretraining** | 55 | 251.6K | 4,574 |
| **Inference Optimization** | 52 | 232.4K | 4,468 |
| **Architecture** | 49 | 226.0K | 4,611 |
| **Evaluation** | 35 | 162.4K | 4,639 |
| **Training Methods** | 32 | 141.8K | 4,432 |
| **Multimodal** | 29 | 141.6K | 4,882 |
| **RL Training** | 24 | 105.6K | 4,399 |
| **LLM Systems** | 18 | 83.2K | 4,622 |
| **Alignment** | 17 | 76.0K | 4,471 |
| **Serving** | 9 | 41.9K | 4,653 |
| **Context Optimization** | 9 | 40.3K | 4,475 |
| **Prompting** | 7 | 29.7K | 4,238 |
| **Low Precision** | 6 | 27.6K | 4,594 |
| **Retrieval** | 3 | 13.9K | 4,637 |

## 🎯 Summarization Task Formulation

We model paper summarization as a conditional text generation task:

Given a research paper P ∈ Papers (represented as PDF or text), we generate a comprehensive summary S = f(P; θ, π) where:
- **f**: language model or human
- **θ**: parameters optimized for technical analysis  
- **π**: Structured prompt encoding our 7-section framework (Executive Summary, Context, Technical Approach, Insights, Experiments, Limitations, Implications)

The objective maximizes informativeness I(S|P) subject to completeness C(S, P) ≥ τ, where a reader gains full paper understanding from S alone without accessing P, with quality threshold τ ensuring: detailed technical mechanisms, cited figures/tables, explicit hyperparameters, and grounded quantitative results.

## 🛠️ Usage

### Process a Single Paper
```bash
uv run python main.py --url https://arxiv.org/abs/XXXX.XXXXX
```

### Batch Process Multiple Papers
```bash
uv run python main.py --urls "url1,url2,url3"
```

### Model
- Default model is **`gpt-5.2`**.
- Override with `--model`, e.g.:

```bash
uv run python main.py --model gpt-5.2 --url https://arxiv.org/abs/XXXX.XXXXX
```

### Prompt
- Default system prompt is **`main_prompt.txt`**.
- Override with `--instructions /path/to/prompt.txt`.

## 📅 Finding Papers

For sources of high-quality papers, check these major conferences and their pages:

### 2025 Conference Calendar

- **COLING 2025** — Jan 19–24, 2025 — Abu Dhabi, UAE
  - Virtual sessions — Jan 27–28, 2025
- **AAAI-25** — Feb 25–Mar 4, 2025 — Philadelphia, PA, USA
- **ICLR 2025** — Apr 24–28, 2025 — Singapore (Singapore EXPO)
  - Conference sessions: Apr 24–26; Workshops: Apr 27–28
- **NAACL 2025** — Apr 29–May 4, 2025 — Albuquerque, NM, USA
- **AISTATS 2025** — May 3–5, 2025 — Mai Khao, Thailand (Splash Beach Resort)
- **MLSys 2025** — May 12–15, 2025 — Santa Clara, CA, USA (Santa Clara Convention Center)
- **ICML 2025** — Jul 13–19, 2025 — Vancouver, BC, Canada (Vancouver Convention Center)
  - Tutorials: Jul 14; Main sessions: Jul 15–17; Workshops: Jul 18–19
- **ACL 2025** — Jul 27–Aug 1, 2025 — Vienna, Austria
- **CoNLL 2025** — Jul 31–Aug 1, 2025 — Vienna, Austria (co-located with ACL)
- **KDD 2025** — Aug 3–7, 2025 — Toronto, ON, Canada (Toronto Convention Centre)
- **EMNLP 2025** — Nov 4–9, 2025 — Suzhou, China
- **NeurIPS 2025** — Nov 30–Dec 7, 2025 — San Diego Convention Center + Hilton Reforma (Mexico City)
  - Tutorials: Dec 2; Main sessions: Dec 3–5; Workshops: Dec 6–7
