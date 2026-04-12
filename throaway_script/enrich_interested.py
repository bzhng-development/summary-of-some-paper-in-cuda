#!/usr/bin/env python3
"""Enrich interested papers that are missing arXiv metadata."""

import sys

sys.path.insert(0, ".")
from daily_papers.hf_daily_papers import fetch_arxiv_metadata
from neon_db import NeonDB

db = NeonDB()

with db.get_conn() as conn, conn.cursor() as cur:
    cur.execute(
        """
        SELECT id FROM "nextjs-ui_paper"
        WHERE interested = 1
          AND id NOT LIKE 'ext:%'
          AND url LIKE '%arxiv%'
          AND (
            categories IS NULL
            OR abstract IS NULL OR abstract = ''
            OR authors IS NULL OR authors = ''
            OR published IS NULL OR published = ''
          )
        """
    )
    missing = [row[0] for row in cur.fetchall()]

print(f"Interested papers missing metadata: {len(missing)}")
if not missing:
    print("Nothing to do.")
    raise SystemExit(0)

for aid in missing:
    print(f"  {aid}")

meta_map = fetch_arxiv_metadata(missing)
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
    print(f"  done {aid} -> {meta.primary_category} ({len(meta.categories)} cats)")

not_found = set(missing) - set(meta_map.keys())
if not_found:
    print(f"\nCould not find on arXiv: {not_found}")

print(f"\nEnriched {enriched}/{len(missing)} interested papers")
