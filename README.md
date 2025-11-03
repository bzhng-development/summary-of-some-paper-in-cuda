# Paper Summaries Knowledge Base

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