"""Shared helpers for Firecrawl CLI scripts."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from loguru import logger
from rich.logging import RichHandler

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

API_BASE = "https://api.firecrawl.dev/v2/crawl"


def require_api_key(env_var: str = "FIRECRAWL_API_KEY") -> str:
    if load_dotenv:
        dotenv_path = os.path.join(os.getcwd(), ".env")
        load_dotenv(dotenv_path=dotenv_path, override=False)
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"{env_var} is not set.")
    return api_key


def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    logging.basicConfig(level=level, handlers=[rich_handler], format="%(message)s")
    rich_logger = logging.getLogger("firecrawl")

    def _sink(message) -> None:
        record = message.record
        rich_logger.log(record["level"].no, record["message"])

    logger.add(_sink, level=level)


def build_headers(api_key: str, *, content_type: bool = False) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if content_type:
        headers["Content-Type"] = "application/json"
    return headers


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as out_f:
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]
