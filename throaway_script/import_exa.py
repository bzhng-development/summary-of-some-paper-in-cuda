#!/usr/bin/env python3
"""Import Exa JSONL entries into papers.db, then enrich via arXiv API."""

import json
import sqlite3
import sys

sys.path.insert(0, ".")

from database import DB_PATH, _migrate_sync, save_paper_sync
from daily_papers.hf_daily_papers import fetch_arxiv_metadata

JSONL = "docs_new/missing-classic-papers-2012-2024.exa.jsonl"


def main():
    # 1) Migrate schema
    _migrate_sync()

    # 2) Load existing IDs
    con = sqlite3.connect(DB_PATH)
    existing = {row[0] for row in con.execute("SELECT id FROM papers").fetchall()}
    con.close()

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
        save_paper_sync(
            aid,
            title=r.get("title"),
            url=f"https://arxiv.org/abs/{aid}",
        )
        existing.add(aid)
        inserted += 1

    print(f"Inserted {inserted} new stub rows")

    # 5) Enrich papers missing arXiv metadata
    con = sqlite3.connect(DB_PATH)
    all_exa_ids = [r["arxiv_id"] for r in entries]
    placeholders = ",".join("?" for _ in all_exa_ids)
    need_enrich = [
        row[0]
        for row in con.execute(
            f"""SELECT id FROM papers
                WHERE id IN ({placeholders})
                  AND (categories IS NULL OR abstract IS NULL OR abstract = ''
                       OR authors IS NULL OR authors = '' OR published IS NULL OR published = '')""",
            all_exa_ids,
        ).fetchall()
    ]
    con.close()

    print(f"Need arXiv enrichment: {len(need_enrich)}")

    if need_enrich:
        meta_map = fetch_arxiv_metadata(need_enrich)
        enriched = 0
        for aid, meta in meta_map.items():
            save_paper_sync(
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
    con = sqlite3.connect(DB_PATH)
    total = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    with_cats = con.execute(
        "SELECT COUNT(*) FROM papers WHERE categories IS NOT NULL"
    ).fetchone()[0]
    con.close()
    print(f"\nDone! DB now has {total} papers ({with_cats} with arXiv metadata)")


if __name__ == "__main__":
    main()
