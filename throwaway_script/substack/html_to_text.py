"""HTML → plain text for Substack podcast-transcript posts.

Preserves speaker turns and inline timestamps. Strips subscribe widgets,
share buttons, paywall blocks, and other UI cruft.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, NavigableString, Tag


_DROP_TAG_NAMES = {
    "script",
    "style",
    "iframe",
    "form",
    "button",
    "svg",
    "noscript",
}

_DROP_CLASS_SUBSTRINGS = (
    "subscribe",
    "share",
    "paywall",
    "footnote-hover",
    "comment-",
    "callout-button",
    "image-link-expand",
)


def _should_drop(tag: Tag) -> bool:
    if tag.name in _DROP_TAG_NAMES:
        return True
    # lxml-built Tags occasionally have attrs=None (notably for DOCTYPE / XML
    # decl nodes); guard before .get("class") to keep the walk going.
    if not getattr(tag, "attrs", None):
        return False
    classes = tag.attrs.get("class") or []
    if not classes:
        return False
    joined = " ".join(classes).lower()
    return any(s in joined for s in _DROP_CLASS_SUBSTRINGS)


def _strip_dom(soup: BeautifulSoup) -> None:
    for tag in list(soup.find_all(True)):
        if _should_drop(tag):
            tag.decompose()


def _block_text(tag: Tag) -> str:
    """Walk a block element and produce its concatenated text.

    Preserves <strong> emphasis markers (Substack uses these for speaker
    labels), keeps anchor text but drops href, replaces <br> with newlines.
    """
    parts: list[str] = []
    for node in tag.descendants:
        if isinstance(node, NavigableString):
            parts.append(str(node))
        elif isinstance(node, Tag):
            if node.name == "br":
                parts.append("\n")
    return "".join(parts).strip()


def html_to_text(html: str) -> str:
    """Convert Substack post HTML to plain text, preserving paragraph + speaker structure."""
    soup = BeautifulSoup(html, "lxml")
    _strip_dom(soup)

    out: list[str] = []
    blocks = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "li", "blockquote", "pre"])
    for b in blocks:
        text = _block_text(b)
        if not text:
            continue
        if b.name in {"h1", "h2", "h3", "h4", "h5"}:
            out.append(f"\n## {text}\n")
        elif b.name == "li":
            out.append(f"  - {text}")
        elif b.name == "blockquote":
            out.append(f"> {text}")
        elif b.name == "pre":
            out.append(f"```\n{text}\n```")
        else:
            out.append(text)

    return "\n\n".join(out).strip() + "\n"


if __name__ == "__main__":
    import sys
    from pathlib import Path

    p = Path(sys.argv[1])
    out = html_to_text(p.read_text(encoding="utf-8"))
    sys.stdout.write(out)
