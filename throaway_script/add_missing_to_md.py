"""Create .md files for the ~796 papers in the regen JSONL that don't yet
have one in paper-graph-ui/src/content/papers.

For each missing arxiv_id:
  - title comes from the arxiv API (authoritative — Neon's `title` column has
    accumulated some wrong arxiv_id↔title mappings over time, e.g. 1204.4894
    is in Neon as "ImageNet Classification..." but the actual PDF is a Tb-W
    physics paper. Confirmed by inspecting paper_text in regen_input_FULL.jsonl.)
    For ext: papers (no arxiv URL), title falls back to Neon since those are
    hand-seeded and the DB title is the only source.
  - category comes from Neon's `category` column
  - pitch is the first ~1-2 sentences of the new s1 (Executive Summary). We
    don't trust Neon's `pitch` column for similar reasons to title.
  - s1-s7 come from the local regen JSONL backups (s7 only for the lucky ~482)
  - url + filename + dir match the convention in
    multi_prompt_pkg/storage.py:save_summary_markdown

Usage:
    DATABASE_URL=... uv run python throaway_script/add_missing_to_md.py \
        --backup-dir local_data/regen_backup_20260525_095404 \
        --content-dir paper-graph-ui/src/content/papers \
        [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psycopg.rows import dict_row

from neon_db import NeonDB, TABLE


ARXIV_FN_RE = re.compile(r"^(\d{4}\.\d{4,5})")
ARXIV_BODY_RE = re.compile(r"\*\*ArXiv:\*\*\s*\[?(\d{4}\.\d{4,5})")
PITCH_FAIL_PREFIXES = (
    "unable to",
    "the provided text",
    "i cannot",
    "i am unable",
    "the text appears to be",
    "the input appears to be",
    "note: the provided",
    "the provided content",
    "the provided document",
    "the provided pdf",
)


def normalize_title_for_filename(title: str, *, max_length: int = 80) -> str:
    cleaned = "".join(ch for ch in title if ord(ch) >= 32)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "-", cleaned).strip("-")[:max_length]


def extract_arxiv_id_from_md(md_path: Path, md_text: str) -> str | None:
    m = ARXIV_FN_RE.match(md_path.stem)
    if m:
        return m.group(1)
    m = ARXIV_BODY_RE.search(md_text[:2000])
    if m:
        return m.group(1)
    return None


def load_section_jsonls(backup_dir: Path) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for n in range(1, 8):
        p = backup_dir / f"regen_output_FULL.s{n}.jsonl"
        d: dict[str, str] = {}
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    aid = rec.get("arxiv_id")
                    text = rec.get("text")
                    if aid and isinstance(text, str) and text.strip():
                        d[aid] = text
        out[n] = d
        print(f"[load] s{n}: {len(d)} records")
    return out


def first_sentence(s1_text: str) -> str:
    """Return the first 1-2 sentences of an s1 body, stripping the heading."""
    body = re.sub(r"^##\s+1\..*?\n+", "", s1_text.strip(), count=1)
    # Match up to two sentence-enders; greedy but bounded.
    m = re.match(r"(.+?[.!?])\s+(?:[A-Z`*_(\d]|$)", body, flags=re.DOTALL)
    if m:
        out = m.group(1).strip()
    else:
        out = body.split("\n", 1)[0].strip()
    return out[:400]


def pitch_looks_broken(p: str | None) -> bool:
    if not p or not p.strip():
        return True
    head = p.lstrip().lower()[:60]
    return any(head.startswith(pref) for pref in PITCH_FAIL_PREFIXES)


def file_id_and_arxiv_line(paper_id: str, url: str | None) -> tuple[str, str]:
    """Mirror multi_prompt_pkg.storage.save_summary_markdown's id/link logic."""
    if paper_id.startswith("ext:"):
        fid = re.sub(r"[^A-Za-z0-9._-]+", "-", paper_id.removeprefix("ext:")).strip("-")
        link = f"**URL:** [{url}]({url})\n" if url else ""
    else:
        fid = paper_id
        link = f"**ArXiv:** [{paper_id}](https://arxiv.org/abs/{paper_id})\n"
    return fid, link


def build_md(
    *,
    title: str,
    arxiv_line: str,
    pitch: str,
    sections: dict[int, str],
) -> str:
    parts: list[str] = [f"# {title}"]
    if arxiv_line:
        parts.append(arxiv_line.rstrip())
    parts.append("## 🎯 Pitch")
    parts.append(pitch.strip())
    parts.append("---")
    for n in range(1, 8):
        if n in sections:
            parts.append(sections[n].rstrip())
    return "\n\n".join(parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup-dir", default="local_data/regen_backup_20260525_095404")
    ap.add_argument("--content-dir", default="paper-graph-ui/src/content/papers")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    backup_dir = Path(args.backup_dir)
    content_dir = Path(args.content_dir)
    if not backup_dir.exists() or not content_dir.exists():
        print("[error] missing backup-dir or content-dir", file=sys.stderr)
        return 1

    by_section = load_section_jsonls(backup_dir)

    # Collect arxiv_ids already represented on disk (filename OR body).
    existing_ids: set[str] = set()
    for p in content_dir.rglob("*.md"):
        if p.name == "index.md":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        aid = extract_arxiv_id_from_md(p, text)
        if aid:
            existing_ids.add(aid)
    print(f"[scan] existing arxiv ids on disk: {len(existing_ids)}")

    missing = sorted(set(by_section[1].keys()) - existing_ids)
    print(f"[scan] missing arxiv ids in regen: {len(missing)}")
    if args.limit:
        missing = missing[: args.limit]
        print(f"[scan] truncated to first {len(missing)} due to --limit")

    db = NeonDB()
    rows_by_id: dict[str, dict] = {}
    CHUNK = 500
    for i in range(0, len(missing), CHUNK):
        batch = missing[i : i + CHUNK]
        with db.get_conn() as c, c.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT id, title, category, url FROM {TABLE} WHERE id = ANY(%s)",
                (batch,),
            )
            for r in cur.fetchall():
                rows_by_id[r["id"]] = r

    # Fetch authoritative titles from arxiv API for the arxiv-prefix subset.
    # ext: papers stay on DB titles since they're not on arxiv at all.
    arxiv_only = [a for a in missing if not a.startswith("ext")]
    print(f"[arxiv-api] fetching titles for {len(arxiv_only)} arxiv ids")
    from daily_papers.hf_daily_papers import fetch_arxiv_metadata
    arxiv_meta = fetch_arxiv_metadata(arxiv_only)
    print(f"[arxiv-api] got {len(arxiv_meta)} responses")

    stats = {
        "wrote": 0,
        "skip_no_db_row": 0,
        "skip_no_title": 0,
        "skip_file_exists": 0,
        "title_from_arxiv": 0,
        "title_from_db": 0,
        "s7_present": 0,
        "s7_missing": 0,
        "new_dirs_created": set(),
    }

    for aid in missing:
        row = rows_by_id.get(aid)
        if not row:
            stats["skip_no_db_row"] += 1
            continue
        category = (row.get("category") or "uncategorized").strip() or "uncategorized"
        url = row.get("url")

        # Title: arxiv API for arxiv-prefix ids (authoritative), DB for ext: ids.
        am = arxiv_meta.get(aid)
        if am and am.title:
            title = am.title.strip()
            stats["title_from_arxiv"] += 1
        else:
            title = (row.get("title") or "").strip()
            if title:
                stats["title_from_db"] += 1
        if not title:
            stats["skip_no_title"] += 1
            continue

        sections = {n: by_section[n].get(aid) for n in range(1, 8) if by_section[n].get(aid)}
        if 1 not in sections:
            stats["skip_no_db_row"] += 1
            continue

        # Pitch from s1's first sentence -- always aligned to content.
        pitch = first_sentence(sections[1])

        if 7 in sections:
            stats["s7_present"] += 1
        else:
            stats["s7_missing"] += 1

        fid, arxiv_line = file_id_and_arxiv_line(aid, url)
        norm = normalize_title_for_filename(title)
        base = f"{fid}-{norm}.md" if norm else f"{fid}.md"

        target_dir = content_dir / category
        target = target_dir / base

        if target.exists():
            stats["skip_file_exists"] += 1
            continue

        body = build_md(
            title=title,
            arxiv_line=arxiv_line,
            pitch=pitch,
            sections=sections,
        )
        if not args.dry_run:
            if not target_dir.exists():
                stats["new_dirs_created"].add(category)
            target_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        stats["wrote"] += 1

    print("\n=== summary ===")
    for k, v in stats.items():
        if k == "new_dirs_created":
            print(f"  {k:20s} {sorted(v)}")
        else:
            print(f"  {k:20s} {v}")
    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
