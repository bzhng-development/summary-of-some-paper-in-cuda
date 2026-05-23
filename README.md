# Paper Summaries Knowledge Base

Visit the site directly.

https://summary-of-some-paper-in-cuda.readthedocs.io/en/latest/llm-systems/2512.02556-DeepSeek-V3.2-PushingtheFrontierofOpenLargeLanguageModels/?h=2512#342-deepseek-sparse-attention-dsa-what-it-is-and-how-it-works

## Run the local UI

Two processes — Python backend + Svelte frontend:

```bash
# terminal 1 — JSON/API backend (reads daily_papers/papers_out/all_scored.json + Neon)
uv run python daily_papers/paper_server.py --port 8787

# terminal 2 — Svelte UI (Vite proxies /api/papers etc. to :8787)
cd svelte-ui && pnpm install && pnpm dev
# open http://localhost:5173
```

Override the backend with `PAPER_SERVER_URL=http://host:port pnpm dev`. More details in `svelte-ui/README.md`.

## Score new papers

```bash
export OPENROUTER_API_KEY=...   # BYOK via OpenRouter → Gemini 3.1 Flash Lite
uv run python daily_papers/hf_daily_papers.py \
  --out-dir daily_papers/papers_out \
  --from 2026-03-01 --to 2026-04-18 \
  --concurrency 16
```

The saver is append-only: `all_scored.json` is re-loaded on every flush and new rows are merged in, so date-scoped re-runs don't truncate prior work.

## Push scored papers into Neon

`paper_server.py` reads `/api/papers` straight from `all_scored.json`, but the "interested" toggle, `add-paper`, and downstream tools go through the Neon `nextjs-ui_paper` table. After a scoring run, mirror the JSON into Neon:

```bash
uv run python sync_db.py --skip-arxiv     # mirror all_scored.json → Neon (fast)
uv run python sync_db.py                  # same, plus enrich from arxiv API
```

Step 5 copies every field — arxiv metadata **and** scoring columns (`score`, `similar_paper`, `score_reason`, `tag_category_v2`, `tag_confidence`, `tag_reason`, `score_source`). Writes go through `NeonDB.batch()` (single connection, commit every 500) so 13 k+ rows don't trip Neon's SSL reaper. Idempotent — safe to re-run.

## Preview:
<img width="2946" height="1592" alt="CleanShot 2026-01-20 at 11 59 25@2x" src="https://github.com/user-attachments/assets/7e352c41-ce56-43e9-b283-21e225663ea6" />



## 📂 Categories

| Category | Papers | Tokens | Avg/Paper |
|----------|--------|--------|-----------|
| **Architecture** | 71 | 361.0K | 5,084 |
| **Pretraining** | 63 | 303.3K | 4,813 |
| **Inference Optimization** | 59 | 279.9K | 4,743 |
| **Evaluation** | 49 | 251.3K | 5,128 |
| **Training Methods** | 46 | 238.9K | 5,193 |
| **LLM Systems** | 37 | 212.3K | 5,736 |
| **Multimodal** | 34 | 178.4K | 5,245 |
| **RL Training** | 31 | 153.6K | 4,955 |
| **Alignment** | 22 | 101.0K | 4,592 |
| **Context Optimization** | 11 | 58.1K | 5,282 |
| **Serving** | 9 | 41.9K | 4,653 |
| **Low Precision** | 8 | 42.7K | 5,334 |
| **Prompting** | 7 | 29.7K | 4,238 |
| **Retrieval** | 4 | 20.5K | 5,120 |

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
