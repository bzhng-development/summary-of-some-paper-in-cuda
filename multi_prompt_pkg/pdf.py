"""PDF fetch and extraction helpers and arxiv URL utilities."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx
import pymupdf
from loguru import logger


def arxiv_id_from_url(url: str) -> str:
    """Extract an ArXiv ID from any ArXiv URL format."""
    path = urlparse(url).path.rstrip("/")
    last_segment = path.split("/")[-1]
    return last_segment.removesuffix(".pdf")


def arxiv_url_to_pdf_url(url: str) -> str:
    """Convert an arxiv ``/abs/`` URL to its ``/pdf/`` counterpart."""
    return url.replace("/abs/", "/pdf/").removesuffix(".pdf")


def download_pdf_bytes(pdf_url: str, *, timeout: float = 60.0) -> bytes:
    """Fetch a PDF and return the raw bytes (native-PDF path)."""
    logger.info("Downloading PDF: {}", pdf_url)
    resp = httpx.get(pdf_url, follow_redirects=True, timeout=timeout)
    resp.raise_for_status()
    logger.info("Downloaded {:.0f} KB", len(resp.content) / 1024)
    return resp.content


def download_and_extract_text(pdf_url: str, *, timeout: float = 60.0) -> str:
    """Fetch a PDF and return its extracted text (text-input path)."""
    pdf_bytes = download_pdf_bytes(pdf_url, timeout=timeout)
    logger.info("Extracting text from {:.0f} KB PDF", len(pdf_bytes) / 1024)
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]
    full_text = "\n\n".join(pages)
    logger.info("Extracted {} chars from {} pages", len(full_text), len(pages))
    return full_text
