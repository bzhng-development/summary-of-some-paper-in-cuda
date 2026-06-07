"""DeepSeek V4-Pro via the cluster vllm (same endpoint T45 uses).

Requires an SSH tunnel to brayden@95.133.253.79:8000 → localhost:8000.
The summarizer script starts/maintains the tunnel itself.
"""

from __future__ import annotations

from openai import OpenAI


CLUSTER_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
DEFAULT_MAX_TOKENS = 32000  # generous; outputs are long-form summaries


def make_client(base_url: str = CLUSTER_BASE_URL) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key="EMPTY",  # vllm doesn't check
        timeout=1200.0,
    )


def call(
    client: OpenAI,
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[str, dict]:
    """Single chat completion. Returns (text, usage_dict)."""
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or ""
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
        "total_tokens": getattr(resp.usage, "total_tokens", None),
        "finish_reason": resp.choices[0].finish_reason,
        "model": resp.model,
    }
    return text, usage
