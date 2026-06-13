#!/usr/bin/env python3
"""Reconcile ``docs/`` as a derived mirror of the canonical paper set.

The canonical, public, viewable paper representation lives in
``paper-graph-ui/src/content/papers/<category>/<id>-<title>.md`` (the "paper
graph"). ``docs/`` is the older/smaller set still consumed by the scorer's
reading-history loader (``ingest/examples.py``), ``cli/sync.py`` stub-backfill,
and the mkdocs site — but it had drifted (501 vs 2208 files) and used an old
mashed-title filename scheme.

This mirrors canon → docs so all three stay current:
  * copy every canon ``<cat>/<file>.md`` into ``docs/<cat>/`` (canon names),
  * drop docs' stale old-named duplicates (same arxiv id, different filename),
  * PRESERVE docs-only papers (arxiv id absent from canon, e.g. classical/
    non-arxiv entries) and mkdocs scaffolding (``index.md`` pages, asset dirs).

Default is a dry run; pass ``--apply`` to write.

Usage (from repo root):
    uv run python -m paper_pipeline.regen.reconcile_docs            # dry run
    uv run python -m paper_pipeline.regen.reconcile_docs --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[3]
CANON_DIR = _REPO_ROOT / "paper-graph-ui" / "src" / "content" / "papers"
DOCS_DIR = _REPO_ROOT / "docs"
_ARXIV_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")


def _arxiv_id(md: Path) -> str | None:
    m = _ARXIV_RE.search(md.read_text(errors="replace")[:600])
    return m.group(1) if m else None


def _paper_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.md") if p.name != "index.md"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")
    ap.add_argument("--canon", type=Path, default=CANON_DIR)
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    args = ap.parse_args(argv)

    canon, docs = args.canon, args.docs
    if not canon.is_dir():
        raise SystemExit(f"canon dir not found: {canon}")

    canon_files = _paper_files(canon)
    # (category, filename) the mirror should contain, and the id->canon-relpath map.
    canon_rel = {p.relative_to(canon) for p in canon_files}
    canon_ids: dict[str, Path] = {}
    for p in canon_files:
        aid = _arxiv_id(p)
        if aid:
            canon_ids[aid] = p.relative_to(canon)

    # Plan copies (canon → docs, same relpath).
    to_copy = [(canon / rel, docs / rel) for rel in sorted(canon_rel, key=str)]
    new_copies = [(s, d) for s, d in to_copy if not d.exists()]

    # Plan stale removals: existing docs paper whose id is in canon but whose
    # relpath isn't a canon relpath → an old-named duplicate of a canon paper.
    to_remove: list[Path] = []
    preserved: list[Path] = []
    for p in _paper_files(docs):
        rel = p.relative_to(docs)
        if rel in canon_rel:
            continue  # exact mirror file (will be overwritten in place)
        aid = _arxiv_id(p)
        if aid and aid in canon_ids:
            to_remove.append(p)  # stale old-named duplicate
        else:
            preserved.append(p)  # docs-only (non-canon) — keep

    new_cats = sorted({rel.parts[0] for rel in canon_rel if not (docs / rel.parts[0]).is_dir()})

    logger.info("canon paper files: {} | docs paper files: {}", len(canon_files), len(_paper_files(docs)))
    logger.info("copy {} files ({} brand-new) → docs/", len(to_copy), len(new_copies))
    logger.info("remove {} stale old-named duplicates", len(to_remove))
    logger.info("preserve {} docs-only (non-canon) papers", len(preserved))
    if new_cats:
        logger.info("new category dirs: {}", ", ".join(new_cats))

    if not args.apply:
        logger.warning("DRY RUN — nothing written. Re-run with --apply to mirror.")
        for _s, d in new_copies[:5]:
            logger.debug("would add: {}", d.relative_to(_REPO_ROOT))
        for p in to_remove[:5]:
            logger.debug("would remove: {}", p.relative_to(_REPO_ROOT))
        return 0

    for src, dst in to_copy:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for p in to_remove:
        p.unlink()
    logger.success(
        "Mirrored canon → docs/: +{} files, -{} stale, {} docs-only kept",
        len(to_copy),
        len(to_remove),
        len(preserved),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
