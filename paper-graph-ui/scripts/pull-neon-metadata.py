#!/usr/bin/env python
# Pull paper metadata from Neon Postgres for every paper present in
# src/lib/graph.generated.json and write the merged result back. Keeps the
# build self-contained — the next `node scripts/build-paper-graph.mjs` will
# pick up the enriched JSON.
#
# Run: `uv run python scripts/pull-neon-metadata.py`
# Requires DATABASE_URL (auto-discovered by neon_db.py).

import json
import sys
from pathlib import Path

# paper_pipeline is editable-installed in the cuda venv (`uv run` resolves up to
# ../pyproject.toml). DATABASE_URL is shell-global; neon_db loads a .env fallback.
from paper_pipeline.core.neon_db import NeonDB
from psycopg.rows import dict_row

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

GRAPH = ROOT / "src/lib/graph.generated.json"
OUT = ROOT / "src/lib/neon-metadata.generated.json"

FIELDS = [
    "id",
    "title",
    "abstract",
    "score",
    "similar_paper",
    "score_reason",
    "tag_category_v2",
    "tag_categories_v2",
    "tag_confidence",
    "tag_reason",
    "authors",
    "affiliations",
    "organization",
    "org_fullname",
    "primary_category",
    "categories",
    "published",
    "cited_by_count",
    "fwci",
    "doi",
    "upvotes",
    "github",
    "github_stars",
    "arxiv_comment",
    "interested",
    "is_only_important_because_of_company",
]


def main():
    if not GRAPH.exists():
        print(f"Run `node scripts/build-paper-graph.mjs` first ({GRAPH} missing)")
        sys.exit(1)
    graph = json.loads(GRAPH.read_text())
    summary_arxiv_ids = sorted(
        {p["arxivId"] for p in graph["papers"] if p.get("arxivId") and p.get("hasSummary", True) is not False}
    )
    summary_arxiv_id_set = set(summary_arxiv_ids)
    print(f"Pulling metadata for {len(summary_arxiv_ids)} summary-backed arxiv IDs…")

    db = NeonDB()
    out = {}
    cols = ", ".join(FIELDS)

    def normalize_row(row, *, has_summary_file, summaryless_scope):
        # psycopg may return jsonb fields as already-parsed dicts/lists,
        # and date-likes as datetime. Normalize to JSON-friendly types.
        clean = {}
        for k, v in row.items():
            if v is None:
                continue
            if hasattr(v, "isoformat"):
                clean[k] = v.isoformat()
            elif isinstance(v, str) and v.startswith(("{", "[")):
                try:
                    clean[k] = json.loads(v)
                except json.JSONDecodeError:
                    clean[k] = v
            else:
                clean[k] = v

        existing = out.get(row["id"], {})
        clean["_has_summary_file"] = bool(existing.get("_has_summary_file")) or has_summary_file
        clean["_summaryless_scope"] = bool(existing.get("_summaryless_scope")) or summaryless_scope
        existing.update(clean)
        out[row["id"]] = existing

    with db.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        # IN chunks to keep the query bounded
        CHUNK = 500
        for i in range(0, len(summary_arxiv_ids), CHUNK):
            batch = summary_arxiv_ids[i : i + CHUNK]
            cur.execute(
                f'SELECT {cols} FROM "nextjs-ui_paper" WHERE id = ANY(%s)',
                (batch,),
            )
            for row in cur.fetchall():
                normalize_row(row, has_summary_file=True, summaryless_scope=False)

        summary_hits = len(out)
        cur.execute(
            f"""
            SELECT {cols}
            FROM "nextjs-ui_paper"
            WHERE (
                is_only_important_because_of_company = true
                OR COALESCE(interested, 0) = 1
            )
              AND (summary IS NULL OR btrim(summary) = '')
            ORDER BY id
            """
        )
        summaryless_hits = 0
        for row in cur.fetchall():
            normalize_row(
                row,
                has_summary_file=row["id"] in summary_arxiv_id_set,
                summaryless_scope=True,
            )
            summaryless_hits += 1

    print(f"  summary hits: {summary_hits} / {len(summary_arxiv_ids)} arxiv IDs found in Neon")
    print(f"  no-summary scope hits: {summaryless_hits}")
    print(f"  union: {len(out)} rows")

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
