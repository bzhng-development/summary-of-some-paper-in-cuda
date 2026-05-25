"""PDF fetch and extraction helpers and arxiv URL utilities."""

import re
import shlex
import shutil
import subprocess
from urllib.parse import urlparse

import httpx
import pymupdf
from loguru import logger


_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


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


def _arxiv_id_if_arxiv_url(pdf_url: str) -> str | None:
    """Return the arxiv_id if pdf_url looks like an arxiv URL, else None."""
    if "arxiv.org" not in pdf_url:
        return None
    aid = arxiv_id_from_url(pdf_url)
    return aid if _ARXIV_ID_RE.match(aid) else None


def extract_text_via_hf_papers(arxiv_id: str, *, timeout: float = 60.0) -> str | None:
    """Try ``hf papers read <arxiv_id>`` for HF's OCR'd markdown.

    HF has already OCR'd arxiv papers into clean markdown — preserving tables,
    equations, and structure better than PyMuPDF's text extraction. Returns
    the markdown on success, or None if ``hf`` is unavailable, the paper isn't
    on HF, or any other failure. Caller must fall back to the PDF path.

    Install the CLI: ``uv pip install -U "huggingface_hub[cli]"``.
    """
    if not shutil.which("hf"):
        return None
    argv = ["hf", "papers", "read", arxiv_id]
    # check=False is intentional: a non-zero rc here is a normal "paper not
    # found / not on HF" signal, and we want the caller to fall back rather
    # than raise. The caller treats `None` as "try the PDF path next".
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("`{}` timed out after {}s", shlex.join(argv), timeout)
        return None
    except OSError as e:
        logger.debug("`{}` failed to spawn: {}", shlex.join(argv), e)
        return None
    if result.returncode != 0 or not result.stdout.strip():
        stderr_excerpt = (result.stderr or "").strip()[:200]
        logger.debug(
            "`{}` returned rc={} (stderr={!r})",
            shlex.join(argv), result.returncode, stderr_excerpt,
        )
        return None
    logger.info("hf papers read {}: got {} chars of markdown", arxiv_id, len(result.stdout))
    return result.stdout


def download_and_extract_text(pdf_url: str, *, timeout: float = 60.0) -> str:
    """Fetch a paper's text. Prefers HF's OCR markdown, falls back to PyMuPDF.

    For arxiv URLs, tries ``hf papers read <arxiv_id>`` first — HF has already
    done OCR and produces cleaner markdown than PyMuPDF's text extraction.
    On any failure (CLI missing, paper not on HF, timeout, empty result) or
    for non-arxiv URLs, falls back to downloading the PDF and extracting with
    PyMuPDF.
    """
    aid = _arxiv_id_if_arxiv_url(pdf_url)
    if aid is not None:
        hf_text = extract_text_via_hf_papers(aid, timeout=timeout)
        if hf_text is not None:
            return hf_text
        logger.info("hf papers unavailable for {}, falling back to PDF", aid)

    pdf_bytes = download_pdf_bytes(pdf_url, timeout=timeout)
    logger.info("Extracting text from {:.0f} KB PDF", len(pdf_bytes) / 1024)
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]
    full_text = "\n\n".join(pages)
    logger.info("Extracted {} chars from {} pages", len(full_text), len(pages))
    return full_text
