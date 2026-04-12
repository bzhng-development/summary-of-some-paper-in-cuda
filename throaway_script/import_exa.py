#!/usr/bin/env python3
"""Import Exa JSONL entries into Neon, then enrich via arXiv API."""

import json
import sys

sys.path.insert(0, ".")

from daily_papers.hf_daily_papers import fetch_arxiv_metadata
from neon_db import NeonDB

JSONL = "docs_new/missing-classic-papers-2012-2024.exa.jsonl"


def main():
    db = NeonDB()

    # 1) Ensure schema
    db.init_schema()

    # 2) Load existing IDs
    existing = db.get_all_ids()

    # 3) Parse Exa JSONL
    entries = []
    for line in open(JSONL):
        r = json.loads(line.strip())
        aid = r.get("arxiv_id")
        if not aid:
            continue
        entries.append(r)

    print(f"Loaded {len(entries)} entries from Exa JSONL")

    # 4) Insert stubs for new papers (title + url)
    inserted = 0
    for r in entries:
        aid = r["arxiv_id"]
        if aid in existing:
            continue
        db.save_paper(
            aid,
            title=r.get("title"),
            url=f"https://arxiv.org/abs/{aid}",
        )
        existing.add(aid)
        inserted += 1

    print(f"Inserted {inserted} new stub rows")

    # 5) Enrich papers missing arXiv metadata
    all_exa_ids = [r["arxiv_id"] for r in entries]
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id FROM "nextjs-ui_paper"
                WHERE id = ANY(%s)
                  AND (categories IS NULL OR abstract IS NULL OR abstract = ''
                       OR authors IS NULL OR authors = '' OR published IS NULL OR published = '')""",
            (all_exa_ids,),
        )
        need_enrich = [row[0] for row in cur.fetchall()]

    print(f"Need arXiv enrichment: {len(need_enrich)}")

    if need_enrich:
        meta_map = fetch_arxiv_metadata(need_enrich)
        enriched = 0
        for aid, meta in meta_map.items():
            db.save_paper(
                aid,
                title=meta.title or None,
                abstract=meta.abstract or None,
                authors=meta.authors or None,
                affiliations=meta.affiliations or None,
                categories=meta.categories or None,
                primary_category=meta.primary_category,
                arxiv_comment=meta.comment,
                published=meta.published,
                journal_ref=meta.journal_ref,
                doi=meta.doi,
            )
            enriched += 1
        print(f"Enriched {enriched}/{len(need_enrich)} papers with arXiv metadata")

    # Summary
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute('SELECT COUNT(*) FROM "nextjs-ui_paper"')
        total = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM "nextjs-ui_paper" WHERE categories IS NOT NULL')
        with_cats = cur.fetchone()[0]
    print(f"\nDone! DB now has {total} papers ({with_cats} with arXiv metadata)")


if __name__ == "__main__":
    main()
