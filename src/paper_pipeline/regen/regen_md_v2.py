"""Refresh paper-graph-ui/src/content/papers/ from a FULL regen run.

Differs from regen_to_md.py:
  - That one only updated existing md sections from s1-s7 jsonls; pitch
    + category came from the existing md or the arxiv API. Used when the
    regen had no pitch/cat phase (the bounce-3 partial state).
  - This one uses the freshly-assembled regen_output_FULL.jsonl which has
    arxiv_id + title + category + pitch + summary + url all from the
    DSV4-Pro run. Applies the new content to every existing md AND moves
    files between category dirs when the new classification differs from
    where the .md was previously filed. New papers (in the jsonl but not
    yet on disk) get created.

Usage:
    uv run python throwaway_script/regen/regen_md_v2.py \\
        --input local_data/regen_backup_<ts>/regen_output_FULL.jsonl \\
        --content-dir paper-graph-ui/src/content/papers \\
        [--dry-run]
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path


ARXIV_FN_RE = re.compile(r"^(\d{4}\.\d{4,5})")
ARXIV_BODY_RE = re.compile(r"\*\*ArXiv:\*\*\s*\[?(\d{4}\.\d{4,5})")
FILE_ID_BODY_RE = re.compile(r"\*\*ArXiv:\*\*\s*\[?(\d{4}\.\d{4,5})|\*\*URL:\*\*\s*\[([^\]]+)\]")


@dataclass(slots=True)
class RegenStats:
    seen: int = 0
    skip_no_title: int = 0
    skip_no_summary: int = 0
    wrote_new: int = 0
    wrote_updated_in_place: int = 0
    wrote_moved_category: int = 0
    deleted_old_path: int = 0
    new_dirs_created: set[str] = field(default_factory=set)

    def display(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            print(f"  {f.name:30s} {sorted(value) if isinstance(value, set) else value}")


def normalize_title_for_filename(title: str, *, max_length: int = 80) -> str:
    cleaned = "".join(ch for ch in title if ord(ch) >= 32)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "-", cleaned).strip("-")[:max_length]


def file_id_and_arxiv_line(paper_id: str, url: str | None) -> tuple[str, str]:
    if paper_id.startswith("ext:"):
        fid = re.sub(r"[^A-Za-z0-9._-]+", "-", paper_id.removeprefix("ext:")).strip("-")
        link = f"**URL:** [{url}]({url})\n" if url else ""
    else:
        fid = paper_id
        link = f"**ArXiv:** [{paper_id}](https://arxiv.org/abs/{paper_id})\n"
    return fid, link


def build_md(*, title: str, arxiv_line: str, pitch: str, summary: str) -> str:
    """Assemble final body. Matches the on-disk format used by paper-graph-ui."""
    parts: list[str] = [f"# {title}"]
    if arxiv_line:
        parts.append(arxiv_line.rstrip())
    parts.append("## 🎯 Pitch")
    parts.append(pitch.strip())
    parts.append("---")
    parts.append(summary.rstrip())
    return "\n\n".join(parts) + "\n"


def index_existing_files(content_dir: Path) -> dict[str, Path]:
    """Map arxiv_id -> current on-disk path (looking at filename + body)."""
    index: dict[str, Path] = {}
    for p in content_dir.rglob("*.md"):
        if p.name == "index.md":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        aid: str | None = None
        if m := ARXIV_FN_RE.match(p.stem):
            aid = m.group(1)
        if aid is None:
            if m := ARXIV_BODY_RE.search(text[:2000]):
                aid = m.group(1)
        if aid is None:
            # ext: papers — try matching the URL line
            # filename like "doi-10.1038-nature14539-Deep-learning.md"
            # without a clean way to recover the original ext: id, we skip
            # matching them by id (they'll appear as "new" in the jsonl and
            # collide on filesystem path — handle via target.exists() check).
            continue
        if aid not in index:
            index[aid] = p
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True, help="regen_output_FULL.jsonl path")
    ap.add_argument("--content-dir", type=Path, default=Path("paper-graph-ui/src/content/papers"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"[error] input missing: {args.input}", file=sys.stderr)
        return 1
    if not args.content_dir.is_dir():
        print(f"[error] content dir missing: {args.content_dir}", file=sys.stderr)
        return 1

    existing = index_existing_files(args.content_dir)
    print(f"[scan] {len(existing)} existing md files indexed by arxiv_id")

    stats = RegenStats()

    for line in args.input.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        stats.seen += 1

        aid = rec["arxiv_id"]
        title = (rec.get("title") or "").strip()
        pitch = (rec.get("pitch") or "").strip()
        summary = (rec.get("summary") or "").strip()
        category = (rec.get("category") or "uncategorized").strip() or "uncategorized"
        url = rec.get("url")

        if not title:
            stats.skip_no_title += 1
            continue
        if not summary:
            stats.skip_no_summary += 1
            continue

        fid, arxiv_line = file_id_and_arxiv_line(aid, url)
        norm = normalize_title_for_filename(title)
        base = f"{fid}-{norm}.md" if norm else f"{fid}.md"
        target_dir = args.content_dir / category
        target_path = target_dir / base

        body = build_md(
            title=title,
            arxiv_line=arxiv_line,
            pitch=pitch,
            summary=summary,
        )

        old_path = existing.get(aid)
        if old_path is not None:
            if old_path == target_path:
                if not args.dry_run:
                    target_path.write_text(body, encoding="utf-8")
                stats.wrote_updated_in_place += 1
            else:
                # Category or filename changed — write new + delete old.
                if not args.dry_run:
                    if not target_dir.is_dir():
                        stats.new_dirs_created.add(category)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(body, encoding="utf-8")
                    old_path.unlink()
                stats.wrote_moved_category += 1
                stats.deleted_old_path += 1
        else:
            # New paper — wasn't on disk yet.
            if not args.dry_run:
                if not target_dir.is_dir():
                    stats.new_dirs_created.add(category)
                target_dir.mkdir(parents=True, exist_ok=True)
                if target_path.exists():
                    # Could collide with an ext: paper whose arxiv_id we
                    # couldn't recover from disk. Overwrite — the regen output
                    # is now the source of truth.
                    pass
                target_path.write_text(body, encoding="utf-8")
            stats.wrote_new += 1

    print("\n=== summary ===")
    stats.display()
    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
