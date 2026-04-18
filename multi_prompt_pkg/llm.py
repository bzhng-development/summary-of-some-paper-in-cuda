"""LLM client construction and retry helpers."""

from __future__ import annotations

import asyncio
import os
import re
import resource
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

from loguru import logger
from openai import AsyncOpenAI

from .config import LOCAL_BASE_URL

T = TypeVar("T")

# ---------------------------------------------------------------------------
# FD limit
# ---------------------------------------------------------------------------


def raise_fd_limit() -> None:
    """Raise the process soft fd limit toward the hard cap.

    macOS ships a 256-file default which is trivially exhausted by a
    high-concurrency async run. We target 65k or the hard cap, whichever is
    smaller. Failures are best-effort and logged, not fatal.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65536 if hard == resource.RLIM_INFINITY else min(hard, 65536)
    if soft >= target:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError) as exc:
        logger.warning("Could not raise fd limit from {}: {}", soft, exc)
        return
    logger.info("Raised fd limit: {} -> {}", soft, target)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMClient:
    """Bundle of an :class:`AsyncOpenAI` client and its per-request extras.

    ``extra_body`` is the provider-specific payload merged into every request
    (SGLang ``chat_template_kwargs`` or OpenRouter provider pinning).
    ``native_pdf`` tells the summarizer whether to send the raw PDF as a file
    attachment rather than extracted text.
    """

    client: AsyncOpenAI
    extra_body: dict[str, Any] | None
    native_pdf: bool = False

    async def aclose(self) -> None:
        await self.client.close()


def build_local_client(
    *,
    base_url: str = LOCAL_BASE_URL,
    api_key: str = "not-needed",
    timeout: float = 1500.0,
    extra_body: dict[str, Any] | None = None,
    preflight: bool = True,
) -> LLMClient:
    """Build a client against a local OpenAI-compatible server (SGLang/vLLM).

    When ``preflight`` is set (default), we issue a blocking ``models.list``
    against the endpoint and raise ``SystemExit(2)`` if it cannot be reached.
    The alternative is a silent 10-attempt exponential backoff on every paper,
    which wastes both real time and any cache the LLM has warmed.
    """
    raise_fd_limit()
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )
    if extra_body is None and base_url == LOCAL_BASE_URL:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    logger.info("Initialized local LLM client at {}", base_url)
    if preflight:
        _preflight_local_server(client, base_url)
    return LLMClient(client=client, extra_body=extra_body, native_pdf=False)


def _preflight_local_server(client: AsyncOpenAI, base_url: str) -> None:
    """Block on ``GET {base_url}/models`` and exit 2 if it doesn't answer.

    Uses a plain sync ``httpx`` call rather than the AsyncOpenAI client because
    this function may be invoked from inside an already-running event loop
    (``main`` → ``asyncio.run(_dispatch)`` → ``build_local_client``).
    """
    _ = client  # kept for future typed probes; silences unused-arg lint
    import httpx

    url = base_url.rstrip("/") + "/models"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        payload = resp.json()
        models = [m.get("id", "?") for m in payload.get("data", [])]
    except Exception as exc:
        logger.error(
            "Local LLM preflight failed at {}: {!r}. "
            "Start SGLang/vLLM/llama.cpp on that port and retry.",
            url,
            exc,
        )
        raise SystemExit(2) from exc
    logger.info("Local LLM preflight OK at {} (models={})", base_url, models)


def build_gemini_client(*, timeout: float = 1500.0) -> LLMClient:
    """Build a client against OpenRouter pinned to Google AI Studio / Gemini.

    The dotenv load is deferred to this call site: only ``--gemini`` runs need
    ``OPENROUTER_API_KEY`` and pulling dotenv at module import would surprise
    other callers.
    """
    from dotenv import load_dotenv

    load_dotenv()
    raise_fd_limit()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )
    extra_body = {"provider": {"only": ["google-ai-studio"], "allow_fallbacks": False}}
    logger.info("Initialized Gemini (OpenRouter) LLM client")
    return LLMClient(client=client, extra_body=extra_body, native_pdf=True)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


_RETRY_PATTERN = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def parse_retry_delay(exc: BaseException) -> float | None:
    """Extract a retry delay (seconds) from a 429-style error message."""
    msg = str(exc)
    match = _RETRY_PATTERN.search(msg)
    if match is not None:
        return float(match.group(1)) + 1.0
    if "429" in msg:
        return 30.0
    return None


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 10,
    label: str = "",
) -> T:
    """Retry an async callable with 429-aware exponential backoff.

    The last attempt lets the exception propagate so callers can decide how to
    surface failures.
    """
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:
            if attempt == max_attempts - 1:
                raise
            wait = parse_retry_delay(exc) or min(2**attempt, 60)
            logger.warning(
                "{}attempt {} failed: {!r}, retrying in {:.0f}s",
                label,
                attempt + 1,
                exc,
                wait,
            )
            await asyncio.sleep(wait)
    # Unreachable: loop either returns or raises.
    raise RuntimeError("retry_async exhausted without a result")
