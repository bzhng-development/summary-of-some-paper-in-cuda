"""Apply local regen JSONL checkpoints back into paper-graph-ui/src/content/papers/*.md.

For each .md file with an extractable arxiv_id that's present in the regen
batch:
  - keep H1 title, ArXiv link, and ## Pitch block as-is (regen didn't redo
    pitch — those run after s7 which only got 482/1274 done before bounce 3)
  - replace s1-s6 with new text from regen_output_FULL.s{1..6}.jsonl
  - s7: use new text if arxiv_id is in regen_output_FULL.s7.jsonl (the lucky
    482), else preserve the old s7 from the existing .md

Usage:
    uv run python throwaway_script/regen/regen_to_md.py \\
        --backup-dir local_data/regen_backup_20260525_095404 \\
        --content-dir paper-graph-ui/src/content/papers \\
        [--dry-run] [--limit 5]
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path


ARXIV_FN_RE = re.compile(r"^(\d{4}\.\d{4,5})")
ARXIV_BODY_RE = re.compile(r"\*\*ArXiv:\*\*\s*\[?(\d{4}\.\d{4,5})")
SECTION_HEAD_RE = re.compile(r"^##\s+(\d+)\.\s", re.MULTILINE)
SECTION_RANGE = range(1, 8)


@dataclass(slots=True)
class RegenStats:
    scanned: int = 0
    no_arxiv_id: int = 0
    not_in_regen: int = 0
    no_sections_parsed: int = 0
    rewrote: int = 0
    s7_new: int = 0
    s7_kept_old: int = 0
    s7_missing: int = 0

    def display(self) -> None:
        for f in fields(self):
            print(f"  {f.name:20s} {getattr(self, f.name)}")


def extract_arxiv_id(md_path: Path, md_text: str) -> str | None:
    """Pull arxiv_id from the filename prefix or the ``**ArXiv:** [id]`` body line."""
    if m := ARXIV_FN_RE.match(md_path.stem):
        return m.group(1)
    if m := ARXIV_BODY_RE.search(md_text[:2000]):
        return m.group(1)
    return None


def load_section_jsonls(backup_dir: Path) -> dict[int, dict[str, str]]:
    """Return ``{section_num: {arxiv_id: section_text}}`` for sections 1..7.

    Missing checkpoint files yield an empty dict for that section, rather than
    failing — the script supports partial regen state by design (s7 was only
    partially done).
    """
    out: dict[int, dict[str, str]] = {}
    for n in SECTION_RANGE:
        path = backup_dir / f"regen_output_FULL.s{n}.jsonl"
        try:
            raw_lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            print(f"[warn] missing {path}", file=sys.stderr)
            out[n] = {}
            continue
        records: dict[str, str] = {}
        for line in raw_lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            aid = rec.get("arxiv_id")
            text = rec.get("text")
            if aid and isinstance(text, str) and text.strip():
                records[aid] = text
        out[n] = records
        print(f"[load] s{n}: {len(records)} records")
    return out


def split_md(md_text: str) -> tuple[str, dict[int, str]]:
    """Return ``(header_block, {section_num: section_text})``.

    ``header_block`` is everything BEFORE the first numbered section heading
    (H1 + ArXiv line + Pitch block + any ``---`` separator). Trailing
    whitespace is normalized to one blank line.

    ``section_text`` includes the heading itself (e.g. ``## 1. Executive
    Summary...``) plus its body up to but not including the next ``## N.``
    heading.
    """
    matches = list(SECTION_HEAD_RE.finditer(md_text))
    if not matches:
        return md_text.rstrip() + "\n", {}
    header = md_text[: matches[0].start()].rstrip() + "\n"
    sections: dict[int, str] = {}
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        sections[n] = md_text[start:end].rstrip() + "\n"
    return header, sections


def assemble_new_md(
    header: str,
    old_sections: dict[int, str],
    new_sections: dict[int, str],
) -> str:
    """Build the new file body from header + new s1-s6 + (new or old s7).

    Returns text ending in a single newline. Sections are joined by one blank
    line between them, matching the existing on-disk format.
    """
    parts: list[str] = [header.rstrip()]
    for n in range(1, 7):
        if text := (new_sections.get(n) or old_sections.get(n)):
            parts.append(text.rstrip())
    if s7 := (new_sections.get(7) or old_sections.get(7)):
        parts.append(s7.rstrip())
    return "\n\n".join(parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backup-dir", type=Path, default=Path("local_data/regen_backup_20260525_095404"))
    ap.add_argument("--content-dir", type=Path, default=Path("paper-graph-ui/src/content/papers"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only update the first N matching files")
    args = ap.parse_args()

    backup_dir: Path = args.backup_dir
    content_dir: Path = args.content_dir

    if not backup_dir.is_dir():
        print(f"[error] backup dir missing: {backup_dir}", file=sys.stderr)
        return 1
    if not content_dir.is_dir():
        print(f"[error] content dir missing: {content_dir}", file=sys.stderr)
        return 1

    by_section = load_section_jsonls(backup_dir)

    md_files = sorted(p for p in content_dir.rglob("*.md") if p.name != "index.md")
    print(f"[scan] {len(md_files)} md files in {content_dir}")

    stats = RegenStats()

    for md in md_files:
        stats.scanned += 1
        text = md.read_text(encoding="utf-8")
        aid = extract_arxiv_id(md, text)
        if aid is None:
            stats.no_arxiv_id += 1
            continue
        # s1 presence proves the paper was in the regen batch.
        if aid not in by_section[1]:
            stats.not_in_regen += 1
            continue
        header, old_sections = split_md(text)
        if not old_sections:
            stats.no_sections_parsed += 1
            continue
        new_sections: dict[int, str] = {n: t for n in SECTION_RANGE if (t := by_section.get(n, {}).get(aid))}
        if 7 in new_sections:
            stats.s7_new += 1
        elif 7 in old_sections:
            stats.s7_kept_old += 1
        else:
            stats.s7_missing += 1
        new_body = assemble_new_md(header, old_sections, new_sections)
        if not args.dry_run:
            md.write_text(new_body, encoding="utf-8")
        stats.rewrote += 1
        if args.limit and stats.rewrote >= args.limit:
            break

    print("\n=== summary ===")
    stats.display()
    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
