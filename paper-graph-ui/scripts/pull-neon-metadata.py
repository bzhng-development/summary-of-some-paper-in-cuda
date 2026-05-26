#!/usr/bin/env python
# Pull paper metadata from Neon Postgres for every paper present in
# src/lib/graph.generated.json and write the merged result back. Keeps the
# build self-contained — the next `node scripts/build-paper-graph.mjs` will
# pick up the enriched JSON.
#
# Run: `uv run python scripts/pull-neon-metadata.py`
# Requires DATABASE_URL (auto-discovered by neon_db.py).

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent  # …/summary-of-some-paper-in-cuda
sys.path.insert(0, str(REPO_ROOT))

# Allow either local .env or the shared one
shared_env = REPO_ROOT.parent.parent / "mine/company-scraper/nextjs-ui/.env"
if shared_env.exists() and "DATABASE_URL" not in os.environ:
    for line in shared_env.read_text().splitlines():
        if line.startswith("DATABASE_URL"):
            _, val = line.split("=", 1)
            os.environ["DATABASE_URL"] = val.strip().strip('"').strip("'")
            break

from neon_db import NeonDB  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

GRAPH = ROOT / "src/lib/graph.generated.json"
OUT = ROOT / "src/lib/neon-metadata.generated.json"

FIELDS = [
    "id", "title", "abstract", "score", "similar_paper", "score_reason",
    "tag_category_v2", "tag_confidence", "tag_reason",
    "authors", "affiliations", "organization", "org_fullname",
    "primary_category", "categories", "published",
    "upvotes", "github", "github_stars", "arxiv_comment",
    "interested", "is_only_important_because_of_company",
]


def main():
    if not GRAPH.exists():
        print(f"Run `node scripts/build-paper-graph.mjs` first ({GRAPH} missing)")
        sys.exit(1)
    graph = json.loads(GRAPH.read_text())
    arxiv_ids = sorted({p["arxivId"] for p in graph["papers"] if p.get("arxivId")})
    if not arxiv_ids:
        print("No arxiv-id papers in graph; nothing to pull")
        return
    print(f"Pulling metadata for {len(arxiv_ids)} arxiv IDs…")

    db = NeonDB()
    out = {}
    cols = ", ".join(FIELDS)
    with db.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        # IN chunks to keep the query bounded
        CHUNK = 500
        for i in range(0, len(arxiv_ids), CHUNK):
            batch = arxiv_ids[i:i + CHUNK]
            cur.execute(
                f'SELECT {cols} FROM "nextjs-ui_paper" WHERE id = ANY(%s)',
                (batch,),
            )
            for row in cur.fetchall():
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
                out[row["id"]] = clean

    print(f"  hit: {len(out)} / {len(arxiv_ids)} arxiv IDs found in Neon")

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
